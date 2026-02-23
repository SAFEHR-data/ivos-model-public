"""CLI script for AR sampling evaluation experiments."""

from functools import partial

import neuralprocesses as nps
import numpy as np
import pandas as pd
import torch
import typer
from loguru import logger
from tqdm import tqdm

import lab as B
from primitivo_model.cli.nps import setup_cuda
from primitivo_model.data.criteria import (
    DailyCriteriaTaskSet,
    create_calibration_plot,
    create_roc_pr_plot,
    evaluate_all_task_predictions,
)
from primitivo_model.evaluation.criteria import compute_classification_metrics
from primitivo_model.mlflow_utils import log_dataframe_to_mlflow, log_plot_to_mlflow, mlflow
from primitivo_model.nps.criteria import collect_sample_predictions
from primitivo_model.nps.data import TaskLoader
from primitivo_model.nps.model import (
    ar_loglike_objective,
    create_model,
    eval_epoch,
    loglike_objective,
)
from primitivo_model.util import convert_str_to_float_or_none, print_banner

app = typer.Typer()


def load_run_parameters(run_id: str) -> dict:
    """Load parameters from the specified MLflow run."""
    run_info = mlflow.get_run(run_id)
    orig_params = run_info.data.params

    # Extract required parameters
    params = {
        "batch_size": int(orig_params["batch_size"]),
        "num_samples": int(orig_params["num_samples"]),
        "length_scale": convert_str_to_float_or_none(orig_params["lengthscale"]),
        "channels": int(orig_params["channels"]),
        "num_layers": int(orig_params["num_layers"]),
        "margin": float(orig_params.get("margin", 0.1)),
        "data_source": orig_params["data_source"],
        "min_task_measurements": int(orig_params.get("min_task_measurements", 0)),
        "forecast_hours": float(orig_params["forecast_hours"]),
        "lookback_hours": float(orig_params["lookback_hours"]),
        "experiment_id": run_info.info.experiment_id,
    }

    logger.info("Successfully loaded parameters from run:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

    return params


def setup_model_and_data(
    run_id: str,
    run_params: dict,
    batch_size: int,
    route: str,
    day_start_hour: int,
    device,
    smoke_test: bool = False,
    tight_criteria: bool = True,
):
    """Setup the model, dataset, and data loader."""
    logger.info(f"Using device: {device}")

    # Load the trained model
    model_location = "model-best.torch"
    logger.info(f"Loading model from run {run_id}...")

    model_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_location}")
    model_data = torch.load(model_path, map_location=device)

    # Create criteria test dataset
    logger.info(f"Creating test dataset with route: {route}")
    if smoke_test:
        logger.info("Running in smoke test mode - using reduced dataset")
    test_set = DailyCriteriaTaskSet(
        route=route,
        data_source=run_params["data_source"],
        min_task_measurements=run_params["min_task_measurements"],
        smoke_test=smoke_test,
        forecast_hours=run_params["forecast_hours"],
        forecast_grid_size=int(
            run_params.get("forecast_grid_size", 3)
        ),  # Default to 3 for backwards compatibility
        lookback_hours=int(run_params["lookback_hours"]),
        day_start_hour=day_start_hour,
        refresh_cache=False,
        tight_criteria=tight_criteria,
    )

    logger.info(f"Test set contains {len(test_set)} tasks")
    logger.info(f"Number of measurements: {test_set.num_measurements}")
    logger.info(f"Measurement names: {test_set.measurement_names}")

    # Create data loader
    test_batcher = TaskLoader(test_set, batch_size=batch_size, device=device, seed=0)

    # Create and load model
    logger.info("Creating model...")
    model = create_model(
        test_set.num_measurements,
        run_params["length_scale"],
        channels=run_params["channels"],
        num_layers=run_params["num_layers"],
        margin=run_params["margin"],
    )
    model.load_state_dict(model_data["weights"])
    model.to(device)
    model.eval()

    # Create initial state
    state = B.create_random_state(torch.float32, seed=0)

    logger.info("Model and data setup complete!")
    return model, test_set, test_batcher, state


def collect_ar_sample_predictions(
    model, test_set, num_samples: int, state, forecast_multiplier: int = 1
):
    """
    Collect AR predictions for all tasks in the test set.

    Args:
        model: The neural process model
        test_set: DailyCriteriaTaskSet instance
        num_samples: Number of samples to generate
        state: Random state for neural processes
        forecast_multiplier: Multiplier for forecast hours in x_test points

    Returns:
        task_preds: Dict with task_id -> {"ft": [...], "yt": [...]}
        state: Updated random state
    """
    model.eval()
    task_preds = {}

    forecast_hours = test_set.split_strategy.forecast_hours
    num_test_points = int(forecast_hours * forecast_multiplier)

    with torch.no_grad():
        for task_id, task in tqdm(test_set.tasks.items(), desc="AR sampling"):
            contexts = task["contexts"]

            # Get the period start time from daily_periods_df using task_id
            period_row = test_set.daily_periods_df[test_set.daily_periods_df.task_name == task_id]
            if len(period_row) == 0:
                raise RuntimeError
            period_start = period_row.iloc[0]["period_start"]
            period_end = period_row.iloc[0]["period_end"]

            # Create test points spanning the forecast period
            x_test = B.to_active_device(
                torch.linspace(period_start, period_end, num_test_points + 1).reshape(1, 1, -1)
            )

            # Cast contexts to proper types
            contexts_cast = []
            for x_ctx, y_ctx in contexts:
                x_ctx_cast = B.cast(torch.float32, x_ctx)
                y_ctx_cast = B.cast(torch.float32, y_ctx)
                contexts_cast.append((x_ctx_cast, y_ctx_cast))

            # Perform AR prediction
            state, _, _, ft_samples, yt_samples = nps.ar_predict(
                state,
                model,
                contexts_cast,
                nps.AggregateInput(*((x_test, i) for i in range(len(test_set.measurement_names)))),
                num_samples=num_samples,
                order="random",
            )

            # Store predictions
            task_preds[task_id] = {"ft": [], "yt": []}
            for f_pred, y_pred in zip(ft_samples, yt_samples):
                task_preds[task_id]["ft"].append(f_pred.squeeze(1, 2))
                task_preds[task_id]["yt"].append(y_pred.squeeze(1, 2))

    return task_preds, state


def evaluate_criteria_predictions(task_preds, test_set, method_name: str):
    """Evaluate criteria predictions for a given set of task predictions."""
    criteria = test_set.get_clinical_criteria(standardised=True)
    measurement_names = test_set.measurement_names

    # Create predictions series
    task_meets_pred = pd.Series(
        evaluate_all_task_predictions(task_preds, measurement_names, criteria)
    )

    # Create results dataframe
    results_df = pd.DataFrame(
        {"meets_criteria": test_set.period_labels, "prob_meets_criteria": task_meets_pred},
        index=test_set.period_labels.index,
    ).dropna()

    logger.info(f"{method_name} - Generated predictions for {len(results_df)} periods")
    logger.info(f"{method_name} - Mean probability: {results_df['prob_meets_criteria'].mean():.3f}")
    logger.info(f"{method_name} - Actual criteria rate: {results_df['meets_criteria'].mean():.3f}")

    return results_df


def evaluate_criteria_predictions_with_criteria(
    task_preds, test_set, method_name: str, tight_criteria: bool
):
    """Evaluate criteria predictions with specified criteria tightness."""
    criteria = test_set.get_clinical_criteria(tight=tight_criteria, standardised=True)
    measurement_names = test_set.measurement_names

    # Create predictions series
    task_meets_pred = pd.Series(
        evaluate_all_task_predictions(task_preds, measurement_names, criteria)
    )

    # Create results dataframe
    results_df = pd.DataFrame(
        {"meets_criteria": test_set.period_labels, "prob_meets_criteria": task_meets_pred},
        index=test_set.period_labels.index,
    ).dropna()

    logger.info(f"{method_name} - Generated predictions for {len(results_df)} periods")
    logger.info(f"{method_name} - Mean probability: {results_df['prob_meets_criteria'].mean():.3f}")
    logger.info(f"{method_name} - Actual criteria rate: {results_df['meets_criteria'].mean():.3f}")

    return results_df


def log_results_to_mlflow(
    results_df: pd.DataFrame,
    method_name: str,
    test_set,
    num_samples: int,
    forecast_multiplier: int,
):
    """Log evaluation results to MLflow."""
    # Get current run ID from active run
    current_run = mlflow.active_run()
    if current_run is None:
        raise ValueError("No active MLflow run found")
    run_id = current_run.info.run_id

    # Compute metrics
    metrics = compute_classification_metrics(results_df, test_set)

    # Log metrics with method prefix
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"{method_name}_{metric_name}", value)

    # Log method-specific parameters
    mlflow.log_param(f"{method_name}_num_samples", num_samples)
    if method_name.startswith("ar"):
        mlflow.log_param(f"{method_name}_forecast_multiplier", forecast_multiplier)

    # Log predictions dataframe
    log_dataframe_to_mlflow(results_df, f"{method_name}_predictions.csv", run_id)

    # Create and log plots
    y_true = np.array(results_df["meets_criteria"])
    y_pred_proba = np.array(results_df["prob_meets_criteria"])

    # ROC/PR plot
    roc_pr_fig = create_roc_pr_plot(y_true, y_pred_proba, method_name)
    log_plot_to_mlflow(roc_pr_fig, f"{method_name}_roc_pr.png")

    # Calibration plot
    cal_fig = create_calibration_plot(y_true, y_pred_proba, method_name)
    log_plot_to_mlflow(cal_fig, f"{method_name}_calibration.png")


@app.command()
def evaluate(
    run_id: str = typer.Option(..., help="MLflow run ID to load model from"),
    route: str = typer.Option(
        "simple-charts-dev", help="Data route (simple-charts-dev or simple-charts)"
    ),
    forecast_multiplier: int = typer.Option(1, help="Multiplier for forecast hours in AR sampling"),
    num_samples: int = typer.Option(200, help="Number of samples for predictions"),
    cuda_device: int = typer.Option(0, help="CUDA device to use"),
    day_start_hour: int = typer.Option(9, help="Day start hour for periods"),
    experiment_name: str = typer.Option("ar_sampling_evaluation", help="MLflow experiment name"),
    smoke_test: bool = typer.Option(False, help="Run smoke test with reduced dataset size"),
    tight_criteria: bool = typer.Option(
        True, help="Use tight (True) or loose (False) clinical criteria bounds"
    ),
):
    """
    Evaluate AR sampling vs normal sampling methods for clinical criteria prediction.

    This script loads a trained model from MLflow, sets up the test dataset, and compares:
    1. Normal sampling method
    2. AR sampling with ft (function) predictions
    3. AR sampling with yt (observation) predictions

    Results are logged to a new MLflow experiment.

    Use --smoke-test for quick validation with reduced dataset and sample size.
    """
    print_banner("# AR Sampling Evaluation")

    # Setup MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")

        # Log experiment parameters
        mlflow.log_param("source_run_id", run_id)
        mlflow.log_param("route", route)
        mlflow.log_param("forecast_multiplier", forecast_multiplier)
        mlflow.log_param("num_samples", num_samples)
        mlflow.log_param("day_start_hour", day_start_hour)
        mlflow.log_param("smoke_test", smoke_test)

        # Load model parameters and setup
        run_params = load_run_parameters(run_id)
        device = setup_cuda(cuda_device)

        model, test_set, test_batcher, state = setup_model_and_data(
            run_id,
            run_params,
            run_params["batch_size"],
            route,
            day_start_hour,
            device,
            smoke_test,
            tight_criteria,
        )

        # Adjust num_samples for smoke test
        if smoke_test and num_samples > 50:
            original_num_samples = num_samples
            num_samples = 50
            logger.info(
                f"Smoke test mode: reducing samples from {original_num_samples} to {num_samples}"
            )

        # Log additional experiment info
        mlflow.log_param("test_set_size", len(test_set))
        mlflow.log_param("num_measurements", test_set.num_measurements)

        # 1. Evaluate normal sampling method (generate predictions once)
        logger.info("=== Evaluating Normal Sampling ===")
        task_preds_normal, state = collect_sample_predictions(
            model, test_batcher, num_samples, state
        )

        # 2. Evaluate AR sampling (generate predictions once)
        logger.info("=== Evaluating AR Sampling ===")
        task_preds_ar, state = collect_ar_sample_predictions(
            model, test_set, num_samples, state, forecast_multiplier
        )

        # Extract AR predictions
        ar_ft_preds = {tid: v["ft"] for tid, v in task_preds_ar.items()}
        ar_yt_preds = {tid: v["yt"] for tid, v in task_preds_ar.items()}

        # Evaluate with both tight and loose criteria
        for criteria_tight in [True, False]:
            criteria_suffix = "tight" if criteria_tight else "loose"
            logger.info(f"=== Evaluating with {criteria_suffix} criteria ===")

            # Evaluate normal sampling
            logger.info(f"--- Normal sampling ({criteria_suffix}) ---")
            results_normal = evaluate_criteria_predictions_with_criteria(
                task_preds_normal, test_set, f"normal_{criteria_suffix}", criteria_tight
            )
            log_results_to_mlflow(
                results_normal,
                f"normal_{criteria_suffix}",
                test_set,
                num_samples,
                forecast_multiplier,
            )

            # Evaluate AR with ft predictions
            logger.info(f"--- AR ft predictions ({criteria_suffix}) ---")
            results_ar_ft = evaluate_criteria_predictions_with_criteria(
                ar_ft_preds, test_set, f"ar_ft_{criteria_suffix}", criteria_tight
            )
            log_results_to_mlflow(
                results_ar_ft,
                f"ar_ft_{criteria_suffix}",
                test_set,
                num_samples,
                forecast_multiplier,
            )

            # Evaluate AR with yt predictions
            logger.info(f"--- AR yt predictions ({criteria_suffix}) ---")
            results_ar_yt = evaluate_criteria_predictions_with_criteria(
                ar_yt_preds, test_set, f"ar_yt_{criteria_suffix}", criteria_tight
            )
            log_results_to_mlflow(
                results_ar_yt,
                f"ar_yt_{criteria_suffix}",
                test_set,
                num_samples,
                forecast_multiplier,
            )

        # 3. Evaluate loss metrics
        logger.info("=== Evaluating Loss Metrics ===")

        # Normal NLL loss
        nll_objective = partial(
            loglike_objective, normalise=True, num_samples=run_params["num_samples"]
        )
        normal_loss, state = eval_epoch(state, model, test_batcher, nll_objective)
        mlflow.log_metric("normal_nll_loss", normal_loss.item())

        # AR NLL loss
        ar_nll_objective = partial(ar_loglike_objective, normalise=True)
        ar_loss, state = eval_epoch(state, model, test_batcher, ar_nll_objective)
        mlflow.log_metric("ar_nll_loss", ar_loss.item())

        # Summary comparison
        logger.info("\n=== Results Summary ===")
        logger.info("Evaluation completed for both tight and loose criteria")
        logger.info("Check MLflow run for detailed metrics and plots")

        logger.info(f"Normal NLL Loss: {normal_loss.item():.4f}")
        logger.info(f"AR NLL Loss: {ar_loss.item():.4f}")

        logger.info(f"\n✅ Evaluation complete! MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    app()
