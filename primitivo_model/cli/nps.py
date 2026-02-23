import copy
from functools import partial

import neuralprocesses.torch as nps
import numpy as np
import torch
import typer
from loguru import logger
from matplotlib import pyplot as plt

import lab.torch as B
from primitivo_model.data.criteria import (
    DailyCriteriaTaskSet,
)
from primitivo_model.data.generator import (
    SampledForecastTaskSet,
)
from primitivo_model.evaluation.criteria import (
    evaluate_criteria_prediction,
    find_optimal_threshold,
)
from primitivo_model.mlflow_utils import (
    log_dataframe_to_mlflow,
    log_plot_to_mlflow,
    mlflow,
)
from primitivo_model.nps.classification import (
    LabelledCriteriaTaskLoader,
    NPClassifier,
    classification_objective,
)
from primitivo_model.nps.analysis import (
    collect_mean_predictions,
    log_raw_mae_summary,
)
from primitivo_model.nps.criteria import (
    CriteriaTaskLoader,
    get_criteria_results_df,
    predict_meets_criteria_exact,
)
from primitivo_model.nps.data import TaskLoader  # Added import
from primitivo_model.nps.model import (
    ar_loglike_objective,
    create_model,
    eval_epoch,
    fast_loglike_objective,
    fit_model,
    mae_objective,
)
from primitivo_model.plots import IVSwitchingAnalysisPlotter, make_and_log_task_plots, plot_task_np
from primitivo_model.util import (
    convert_str_to_float_or_none,
    print_banner,
)


def setup_cuda(cuda_device):
    # Set up device
    if cuda_device >= 0 and torch.cuda.is_available():
        logger.info(f"Using CUDA device {cuda_device}")
        device = torch.device(f"cuda:{cuda_device}")
    else:
        logger.info("Using CPU for computation")
        device = torch.device("cpu")
    B.set_global_device(device)
    return device


app = typer.Typer()


def _run_training(
    cuda_device: int,
    route: str,
    data_source: str,
    learning_rate: float,
    num_epochs: int,
    patience: int,
    batch_size: int,
    epoch_size: int,
    smoke_test: bool,
    forecast_hours: int,
    forecast_sparse_only: bool,
    lookback_hours: float,
    random_seed: int,
    epsilon: float,
    channels: int,
    num_layers: int,
    normalise_objective: bool,
    experiment_name: str,
    lengthscale: float,
    margin: float,
    min_task_measurements: int,
    train_tasks_per_day: float,
    val_epoch_size: int,
    refresh_cache: bool,
    dataset_fraction: float,
    use_lr_scheduler: bool,
) -> str:
    """
    Internal function that performs the actual training.
    Returns the run_id and allows training objects to be garbage collected.
    """
    print_banner("## 🤖 Training 🎓")

    device = setup_cuda(cuda_device)
    B.set_random_seed(random_seed)
    state = B.create_random_state(torch.float32, seed=random_seed)

    mlflow.set_experiment(experiment_name)

    # if smoke_test set everything to values that will run quickly
    if smoke_test:
        num_epochs = 2
        batch_size = 2
        epoch_size = 4

    # Start MLflow run
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        # Log all parameters except mlflow ones
        params = copy.copy(locals())
        del params["experiment_name"]
        # this gets modified so we log later
        del params["lengthscale"]
        mlflow.log_params(params)
        mlflow.log_param("model_type", "neural_process")

        shared_task_params = {
            "route": route,
            "smoke_test": smoke_test,
            "data_source": data_source,  # Add data_source parameter
            "min_task_measurements": min_task_measurements,
            "tasks_per_day": train_tasks_per_day,
            "refresh_cache": refresh_cache,
            "lookback_hours": lookback_hours,
            "forecast_hours": forecast_hours,
            "sparse_only": forecast_sparse_only,
        }

        train_tasks = SampledForecastTaskSet(
            **shared_task_params,
            subset="train",
        )

        train_batcher = TaskLoader(
            train_tasks,
            batch_size=batch_size,
            device=device,
            epoch_size=epoch_size,
            seed=random_seed,
            dataset_fraction=dataset_fraction,
        )
        mlflow.log_param("num_train_tasks", len(train_tasks.tasks))

        val_tasks = SampledForecastTaskSet(
            **shared_task_params,
            subset="val",
        )

        val_batcher = TaskLoader(
            val_tasks,
            epoch_size=val_epoch_size if not smoke_test else 2**4,
            batch_size=batch_size,
            device=device,
            seed=random_seed,
        )

        min_timestep = train_tasks.get_min_timestep()
        mlflow.log_param("min_timestep", min_timestep)
        logger.info(f"Minimum timestep {min_timestep:.2f} hours ({min_timestep * 60:.0f} mins)")
        if lengthscale is None:
            logger.info(
                f"Manually overriding lengthscale to {min_timestep:.2f} hours ({min_timestep * 60:.0f} mins)"
            )
            lengthscale = min_timestep

        mlflow.log_param("lengthscale", lengthscale)

        # Create model
        model = create_model(
            train_tasks.num_measurements,
            lengthscale,
            channels=channels,
            num_layers=num_layers,
            margin=margin,
        ).to(device)

        mlflow.log_param("num_params", nps.num_params(model))

        objective = partial(fast_loglike_objective, normalise=normalise_objective)

        # Start training
        logger.info(f"Starting training for {num_epochs} epochs")
        model, state = fit_model(
            state,
            model,
            train_batcher,  # Changed from train_gen
            val_batcher,  # Changed from val_gen
            objective,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            epsilon=epsilon,
            use_lr_scheduler=use_lr_scheduler,
        )

        logger.info(f"Training completed and model artifacts saved in MLflow run: {run_id}")

        mae = partial(mae_objective)
        train_mae, state = eval_epoch(state, model, train_batcher, mae)  # Changed from test_gen
        # Log evaluation metrics
        mlflow.log_metric("train_mae_last", train_mae)

    # Return the run_id so evaluation can be run after training objects are GC'd
    return run_id


@app.command(name="train", short_help="Train a Neural Process model on time series data")
def train(
    cuda_device: int = typer.Option(0, help="CUDA device ID to use for training (-1 for CPU)"),
    route: str = typer.Option("mock", help="Route to use for the data"),
    data_source: str = typer.Option("radix", help="Data source to use ('radix' or 'mimic4')"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate for the optimizer"),
    num_epochs: int = typer.Option(100, help="Number of training epochs"),
    patience: int = typer.Option(5, help="Patience for early stopping"),
    batch_size: int = typer.Option(16, help="Training batch size"),
    epoch_size: int = typer.Option(2**14, help="Number of tasks per epoch"),
    smoke_test: bool = typer.Option(
        False,
        help="Run a smoke test with a small dataset and few epochs for quick validation",
    ),
    forecast_hours: float = typer.Option(
        12.0, help="Hours to forecast for each task (encounter) in validation"
    ),
    forecast_grid_size: int = typer.Option(3, help="Number of grid points in forecast window"),
    forecast_sparse_only: bool = typer.Option(
        False,
        help="Whether to forecast only the sparse target points in validation",
    ),
    lookback_hours: float = typer.Option(
        48, help="Number of hours to look back for each task (encounter)"
    ),
    random_seed: int = typer.Option(1, help="Random seed for reproducibility"),
    epsilon: float = typer.Option(1e-6, help="reg strength for set conv"),
    channels: int = typer.Option(64, help="Number of channels in each UNet layer"),
    num_layers: int = typer.Option(6, help="Number of UNet layers"),
    normalise_objective: bool = typer.Option(
        True, help="Whether to normalise the log-likelihood objective"
    ),
    experiment_name: str = typer.Option("neural_process_training", help="MLflow experiment name"),
    lengthscale: float = typer.Option(
        1.0, help="Optional lengthscale parameter for the ConvCNP model"
    ),
    margin: float = typer.Option(1.0, help="Margin parameter for the ConvCNP model"),
    min_task_measurements: int = typer.Option(10, help="Minimum number of measurements per task"),
    train_tasks_per_day: float = typer.Option(
        1.0, help="Number of tasks to generate per day of encounter data for training"
    ),
    compute_ar_loglike: bool = typer.Option(
        False, help="Whether to compute autoregressive log-likelihood during evaluation"
    ),
    val_epoch_size: int = typer.Option(2**12, help="Number of tasks per validation epoch"),
    refresh_cache: bool = typer.Option(False, help="Refresh the task cache"),
    dataset_fraction: float = typer.Option(
        1.0, help="Fraction of training tasks to use (0.0 to 1.0)"
    ),
    use_lr_scheduler: bool = typer.Option(
        False, help="Use learning rate scheduler with warmup and cosine annealing"
    ),
    tight_criteria: bool = typer.Option(
        True, help="Use tight (True) or loose (False) clinical criteria bounds"
    ),
):
    """
    Train a Neural Process model and run evaluation.

    This command orchestrates the full training pipeline:
    1. Trains the model (via _run_training)
    2. Runs evaluation on test set (via evaluate)
    3. Runs criteria-based evaluation (via evaluate_criteria)

    Training objects are garbage collected between steps to minimize memory usage.
    """
    # Run training - this returns the run_id and allows training objects to be GC'd
    run_id = _run_training(
        cuda_device=cuda_device,
        route=route,
        data_source=data_source,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        patience=patience,
        batch_size=batch_size,
        epoch_size=epoch_size,
        smoke_test=smoke_test,
        forecast_hours=forecast_hours,
        forecast_sparse_only=forecast_sparse_only,
        lookback_hours=lookback_hours,
        random_seed=random_seed,
        epsilon=epsilon,
        channels=channels,
        num_layers=num_layers,
        normalise_objective=normalise_objective,
        experiment_name=experiment_name,
        lengthscale=lengthscale,
        margin=margin,
        min_task_measurements=min_task_measurements,
        train_tasks_per_day=train_tasks_per_day,
        val_epoch_size=val_epoch_size,
        refresh_cache=refresh_cache,
        dataset_fraction=dataset_fraction,
        use_lr_scheduler=use_lr_scheduler,
    )

    evaluate(
        run_id,
        cuda_device,
        load_last=False,
        num_tasks_to_plot=10,
        smoke_test=smoke_test,
        compute_ar_loglike=compute_ar_loglike,
        refresh_cache=refresh_cache,
    )

    evaluate_criteria(
        run_id,
        cuda_device,
        num_tasks_to_plot=10,
        smoke_test=smoke_test,
        day_start_hour=9,
        create_nested_run=False,
        refresh_cache=refresh_cache,
        tight_criteria=tight_criteria,
        forecast_grid_size=forecast_grid_size,
    )


@app.command(name="evaluate", short_help="Evaluate a trained Neural Process model on test data")
def evaluate(
    run_id: str = typer.Argument(..., help="MLflow run ID of the trained model to evaluate"),
    cuda_device: int = typer.Option(0, help="CUDA device ID to use for evaluation (-1 for CPU)"),
    num_tasks_to_plot: int = typer.Option(5, help="Number of test tasks to plot predictions for"),
    load_last: bool = typer.Option(
        False,
        help="Load the last model from the run. If False, load the best model",
    ),
    smoke_test: bool = typer.Option(
        False,
        help="Run a smoke test with a small dataset and few epochs for quick validation",
    ),
    compute_ar_loglike: bool = typer.Option(
        False, help="Whether to compute autoregressive log-likelihood during evaluation"
    ),
    refresh_cache: bool = typer.Option(False, help="Refresh the task cache"),
):
    print_banner("## 📉 Evaluation 📈")
    # Get run information for resuming the previous run
    run_info = mlflow.get_run(run_id)
    experiment_id = run_info.info.experiment_id
    device = setup_cuda(cuda_device)

    orig_params = run_info.data.params

    # Extract required parameters - will raise KeyError if any are missing
    try:
        batch_size = int(orig_params["batch_size"])
        forecast_hours = float(orig_params["forecast_hours"])
        data_source = orig_params["data_source"]
        sparse_only = orig_params["forecast_sparse_only"].lower() == "true"
        normalise_objective = orig_params["normalise_objective"].lower() == "true"
        length_scale = convert_str_to_float_or_none(orig_params["lengthscale"])
        channels = int(orig_params["channels"])
        num_layers = int(orig_params["num_layers"])
        route = orig_params["route"]
        lookback_hours = convert_str_to_float_or_none(orig_params["lookback_hours"])

        # Extract minimum task measurements parameter, default to 0 if not found for backward compatibility
        min_task_measurements = int(orig_params.get("min_task_measurements", 0))

        # Extract margin parameter, default to 0.1 if not found for backward compatibility
        margin = float(orig_params.get("margin", 0.1))

    except KeyError as e:
        missing_param = str(e).strip("'")
        raise typer.BadParameter(f"Required parameter '{missing_param}' not found in original run")

    # Continue with the same run to add evaluation metrics
    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
        logger.info(f"Loading model from run {run_id}")

        # Load the model state dict
        if load_last:
            model_location = "model-last.torch"
            logger.info("Loading last model")
        else:
            model_location = "model-best.torch"
            logger.info("Loading best model")

        model_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_location}")

        model_data = torch.load(model_path, map_location=device)

        # Create test data generator
        test_tasks = SampledForecastTaskSet(
            forecast_hours=forecast_hours,
            lookback_hours=lookback_hours,
            subset="test",
            route=route,
            sparse_only=sparse_only,
            smoke_test=smoke_test,
            data_source=data_source,
            min_task_measurements=min_task_measurements,
            tasks_per_day=0.5,  # Fixed to 0.5 for test tasks
            refresh_cache=refresh_cache,
        )
        test_batcher = TaskLoader(test_tasks, batch_size=batch_size, device=device, seed=0)

        model = create_model(
            test_tasks.num_measurements,  # Changed from test_gen
            length_scale,
            channels=channels,
            num_layers=num_layers,
            margin=margin,
        )
        model.load_state_dict(model_data["weights"])
        model.to(device)
        model.eval()
        # Create objective function

        # Create initial state
        state = B.create_random_state(torch.float32, seed=0)

        # Evaluate the model\
        nll_objective = partial(fast_loglike_objective, normalise=normalise_objective)

        test_loss, state = eval_epoch(
            state, model, test_batcher, nll_objective
        )  # Changed from test_gen
        # Log evaluation metrics
        mlflow.log_metric(f"test_loss{'_last' if load_last else ''}", test_loss)
        logger.info(f"Test loss: {test_loss}")

        # Evaluate the model with AR log-likelihood (optional)
        if compute_ar_loglike:
            ar_nll_objective = partial(ar_loglike_objective, normalise=normalise_objective)

            ar_test_loss, state = eval_epoch(
                state, model, test_batcher, ar_nll_objective
            )  # Changed from test_gen
            # Log evaluation metrics
            mlflow.log_metric(f"ar_test_loss{'_last' if load_last else ''}", ar_test_loss)
            logger.info(f"AR Test loss: {ar_test_loss}")
        else:
            logger.info("Skipping AR log-likelihood computation")

        mae = partial(mae_objective)

        test_mae, state = eval_epoch(state, model, test_batcher, mae)  # Changed from test_gen
        # Log evaluation metrics
        mlflow.log_metric(f"test_mae{'_last' if load_last else ''}", test_mae)
        logger.info(f"Test mae: {test_mae}")

        task_preds, state = collect_mean_predictions(model, test_batcher, state)
        log_raw_mae_summary(task_preds, test_tasks, run.info.run_id)

        # zoomed to full task
        plot_task_fn = partial(plot_task_np, model, hours_back=lookback_hours)
        # Generate and log prediction plots
        make_and_log_task_plots(
            plot_task_fn, num_tasks_to_plot if not smoke_test else 1, test_tasks, run.info.run_id
        )
        # zoomed to pred horizon
        plot_task_fn = partial(plot_task_np, model, hours_back=forecast_hours)
        make_and_log_task_plots(
            plot_task_fn,
            num_tasks_to_plot if not smoke_test else 1,
            test_tasks,
            run.info.run_id,
            fig_name="predictions_zoomed_task",
        )

        return test_loss


@app.command(
    name="evaluate-criteria",
    short_help="Evaluate a trained Neural Process model on criteria prediction",
)
def evaluate_criteria(
    run_id: str = typer.Argument(..., help="MLflow run ID of the trained model to evaluate"),
    cuda_device: int = typer.Option(0, help="CUDA device ID to use for evaluation (-1 for CPU)"),
    num_tasks_to_plot: int = typer.Option(5, help="Number of test tasks to plot predictions for"),
    smoke_test: bool = typer.Option(
        False,
        help="Run a smoke test with a small dataset and few epochs for quick validation",
    ),
    day_start_hour: int = typer.Option(9, help="Hour of day when daily periods start (0-23)"),
    refresh_cache: bool = typer.Option(False, help="Refresh the task cache"),
    tight_criteria: bool = typer.Option(
        True, help="Use tight (True) or loose (False) clinical criteria bounds"
    ),
    forecast_grid_size: int = typer.Option(3, help="Number of grid points in forecast window"),
    create_nested_run: bool = typer.Option(
        False,
        help="Create a nested run for this evaluation to avoid overwriting metrics from previous evaluations with different parameters",
    ),
):
    print_banner("## 🏥 Criteria Evaluation 📊")
    eval_params = copy.copy(locals())
    # Get run information for resuming the previous run
    run_info = mlflow.get_run(run_id)
    experiment_id = run_info.info.experiment_id
    device = setup_cuda(cuda_device)

    orig_params = run_info.data.params

    # Extract required parameters
    try:
        batch_size = int(orig_params["batch_size"])
        length_scale = convert_str_to_float_or_none(orig_params["lengthscale"])
        channels = int(orig_params["channels"])
        num_layers = int(orig_params["num_layers"])
        route = orig_params["route"]
        margin = float(orig_params.get("margin", 0.1))
        data_source = orig_params["data_source"]
        min_task_measurements = int(orig_params.get("min_task_measurements", 0))
        forecast_hours = float(orig_params["forecast_hours"])
        lookback_hours = float(orig_params["lookback_hours"])
    except KeyError as e:
        missing_param = str(e).strip("'")
        raise typer.BadParameter(f"Required parameter '{missing_param}' not found in original run")

    # Start the parent run context
    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id):
        # Optionally create a nested run inside
        if create_nested_run:
            nested_run = mlflow.start_run(
                experiment_id=experiment_id,
                nested=True,
            )

            for param_key, param_value in eval_params.items():
                if not param_key == "run_id":
                    mlflow.log_param(param_key, param_value)

            # # Log all parent run parameters to the nested run
            for param_key, param_value in orig_params.items():
                # check param not alreay logged
                if param_key not in eval_params.keys():
                    mlflow.log_param(param_key, param_value)

            # Get the active run ID for logging
            active_run_id = mlflow.active_run().info.run_id
        else:
            # Use a dummy context manager that does nothing
            from contextlib import nullcontext

            nested_run = nullcontext()
            active_run_id = run_id

        with nested_run:
            logger.info(f"Loading model from run {run_id} for criteria evaluation")

            # Load the best model
            model_location = "model-best.torch"
            logger.info("Loading best model")

            model_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_location}")
            model_data = torch.load(model_path, map_location=device)

            # Create criteria test dataset using original parameters
            test_set = DailyCriteriaTaskSet(
                route=route,
                data_source=data_source,
                min_task_measurements=min_task_measurements,
                smoke_test=smoke_test,
                forecast_hours=forecast_hours,
                forecast_grid_size=forecast_grid_size,
                lookback_hours=int(lookback_hours),
                day_start_hour=day_start_hour,
                refresh_cache=refresh_cache,
                tight_criteria=tight_criteria,
            )
            test_batcher = CriteriaTaskLoader(
                test_set, batch_size=batch_size, device=device, seed=0
            )

            # Create and load model
            model = create_model(
                test_set.num_measurements,
                length_scale,
                channels=channels,
                num_layers=num_layers,
                margin=margin,
            )
            model.load_state_dict(model_data["weights"])
            model.to(device)
            model.eval()

            # Create initial state
            state = B.create_random_state(torch.float32, seed=0)

            # Make probabilistic predictions
            logger.info("Making probabilistic criteria predictions...")
            task_meets_pred, state = predict_meets_criteria_exact(state, model, test_batcher)

            logger.info("Logging probabilistic criteria predictions...")
            # Prepare results dataframes
            model_results = get_criteria_results_df(task_meets_pred, test_set)
            log_dataframe_to_mlflow(
                model_results, f"test_criteria_preds_{active_run_id}.csv", active_run_id
            )

            # Use unified evaluation function
            model_metrics = evaluate_criteria_prediction(
                results_df=model_results,
                test_set=test_set,
                model_name="Model",
                plot_prefix="model",
                log_filename="criteria_metrics.csv",
                run_id=active_run_id,
            )

            logger.info("Loading validation data to find optimal threshold...")
            val_set = DailyCriteriaTaskSet(
                route=route,
                data_source=data_source,
                min_task_measurements=min_task_measurements,
                smoke_test=smoke_test,
                forecast_hours=forecast_hours,
                forecast_grid_size=forecast_grid_size,
                lookback_hours=int(lookback_hours),
                day_start_hour=day_start_hour,
                tight_criteria=tight_criteria,
                refresh_cache=refresh_cache,
                subset="val",
            )
            val_batcher = CriteriaTaskLoader(val_set, batch_size=batch_size, device=device, seed=0)
            val_task_meets_pred, state = predict_meets_criteria_exact(state, model, val_batcher)
            val_results = get_criteria_results_df(val_task_meets_pred, val_set)
            optimal_threshold = find_optimal_threshold(
                np.array(val_results["meets_criteria"].values),
                np.array(val_results["prob_meets_criteria"].values),
            )
            mlflow.log_metric("optimal_threshold", optimal_threshold)

            # Select random task IDs for IV switching plots
            task_ids = np.array(model_results.dropna(subset=["meets_criteria"]).index.tolist())
            np.random.default_rng(0).shuffle(task_ids)
            random_task_ids = task_ids[: min(num_tasks_to_plot if not smoke_test else 2, len(task_ids))]

            # Generate IV switching analysis plots for selected patients
            logger.info("Generating IV switching analysis plots...")

            # Join results with period data and IV labels
            joined_periods_preds_labels = test_set.daily_periods_df.join(
                model_results, on="task_name"
            )
            joined_periods_preds_labels = joined_periods_preds_labels.loc[
                joined_periods_preds_labels.meets_criteria.notna()
            ]
            # Extract patient IDs from the already selected task IDs
            selected_patients = []
            for task_id in random_task_ids:
                # Get the patient ID corresponding to this task
                patient_periods = joined_periods_preds_labels[
                    joined_periods_preds_labels.task_name == task_id
                ]
                if not patient_periods.empty:
                    patient_id = patient_periods["pat_enc_csn_id"].iloc[0]
                    if patient_id not in selected_patients:
                        selected_patients.append(patient_id)

            plotter = IVSwitchingAnalysisPlotter()

            for i, patient_id in enumerate(selected_patients):
                fig = plotter.plot_iv_switching_analysis(
                    patient_id, joined_periods_preds_labels, test_set
                )

                log_plot_to_mlflow(fig, f"iv_switching_analysis_patient_{i + 1}_{patient_id}.png")
                plt.close(fig)  # Close to free memory

            return model_metrics["auroc"]


@app.command(
    name="train-head",
    short_help="Train a classification head on a pre-trained Neural Process model",
)
def train_classification_head(
    base_run_id: str = typer.Argument(
        ..., help="MLflow run ID of the pre-trained NP model to use as base"
    ),
    cuda_device: int = typer.Option(0, help="CUDA device ID to use for training (-1 for CPU)"),
    learning_rate: float = typer.Option(1e-4, help="Learning rate for the optimizer"),
    num_epochs: int = typer.Option(100, help="Maximum number of training epochs"),
    patience: int = typer.Option(10, help="Patience for early stopping"),
    batch_size: int = typer.Option(
        512, help="Batch size (overrides base model batch size if provided)"
    ),
    smoke_test: bool = typer.Option(
        False, help="Run a smoke test with a small dataset for quick validation"
    ),
    day_start_hour: int = typer.Option(9, help="Hour of day when daily periods start (0-23)"),
    refresh_cache: bool = typer.Option(False, help="Refresh the task cache"),
    tight_criteria: bool = typer.Option(
        True, help="Use tight (True) or loose (False) clinical criteria bounds"
    ),
    hidden_dims: str = typer.Option(
        "256,128", help="Comma-separated list of hidden layer dimensions"
    ),
    pooling: str = typer.Option(
        "last",
        help="Temporal pooling method: 'last', 'avg', 'max', or 'attention'",
    ),
    dropout: float = typer.Option(0.3, help="Dropout probability"),
    freeze_np: bool = typer.Option(True, help="Freeze the Neural Process weights during training"),
    experiment_name: str = typer.Option(
        "np_classification_training", help="MLflow experiment name"
    ),
    use_lr_scheduler: bool = typer.Option(
        False, help="Use learning rate scheduler with warmup and cosine annealing"
    ),
    forecast_grid_size: int = typer.Option(3, help="Number of grid points in forecast window"),
):
    """
    Train a classification head on a pre-trained Neural Process model.

    This command:
    1. Loads a pre-trained NP model from MLflow
    2. Creates a classification head with specified architecture
    3. Trains using the existing fit_model() function with early stopping
    4. Evaluates on validation and test sets using existing evaluation infrastructure
    5. Logs all metrics and artifacts to MLflow
    """
    print_banner("## 🧠 Classification Head Training 🎯")

    # Capture all local variables for logging (like in nps.py)
    cli_params = locals().copy()

    # Parse hidden dimensions
    hidden_dims_list = [int(x.strip()) for x in hidden_dims.split(",")]

    # Set up device
    device = setup_cuda(cuda_device)

    # Get base run information
    base_run_info = mlflow.get_run(base_run_id)
    base_params = base_run_info.data.params

    # Extract required parameters from base run
    try:
        length_scale = convert_str_to_float_or_none(base_params["lengthscale"])
        channels = int(base_params["channels"])
        num_layers = int(base_params["num_layers"])
        route = base_params["route"]
        margin = float(base_params.get("margin", 0.1))
        data_source = base_params["data_source"]
        min_task_measurements = int(base_params.get("min_task_measurements", 0))
        forecast_hours = float(base_params["forecast_hours"])
        lookback_hours = float(base_params["lookback_hours"])
    except KeyError as e:
        missing_param = str(e).strip("'")
        raise typer.BadParameter(
            f"Required parameter '{missing_param}' not found in base run {base_run_id}"
        )

    # Use provided batch_size
    actual_batch_size = batch_size

    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")

        # Log all CLI parameters using locals (excluding non-serializable objects)
        for key, value in cli_params.items():
            if key not in ["app", "typer"] and not callable(value):
                mlflow.log_param(key, value)

        # Log parsed hidden dims
        mlflow.log_param("hidden_dims_list", str(hidden_dims_list))

        # Log base model parameters with prefix
        for key, value in base_params.items():
            mlflow.log_param(f"base_{key}", value)

        # Load pre-trained NP model
        logger.info(f"Loading pre-trained model from run {base_run_id}...")
        from mlflow.artifacts import download_artifacts

        model_path = download_artifacts(f"runs:/{base_run_id}/model-best.torch")
        model_data = torch.load(model_path, map_location=device)

        # Create datasets
        logger.info("Creating datasets...")
        train_set = DailyCriteriaTaskSet(
            route=route,
            data_source=data_source,
            min_task_measurements=min_task_measurements,
            smoke_test=smoke_test,
            forecast_hours=forecast_hours,
            forecast_grid_size=forecast_grid_size,
            lookback_hours=int(lookback_hours),
            day_start_hour=day_start_hour,
            refresh_cache=refresh_cache,
            tight_criteria=tight_criteria,
            subset="train",
        )

        val_set = DailyCriteriaTaskSet(
            route=route,
            data_source=data_source,
            min_task_measurements=min_task_measurements,
            smoke_test=smoke_test,
            forecast_hours=forecast_hours,
            forecast_grid_size=forecast_grid_size,
            lookback_hours=int(lookback_hours),
            day_start_hour=day_start_hour,
            refresh_cache=refresh_cache,
            tight_criteria=tight_criteria,
            subset="val",
        )

        test_set = DailyCriteriaTaskSet(
            route=route,
            data_source=data_source,
            min_task_measurements=min_task_measurements,
            smoke_test=smoke_test,
            forecast_hours=forecast_hours,
            forecast_grid_size=forecast_grid_size,
            lookback_hours=int(lookback_hours),
            day_start_hour=day_start_hour,
            refresh_cache=refresh_cache,
            tight_criteria=tight_criteria,
            subset="test",
        )

        logger.info(f"Train set: {len(train_set)} tasks")
        logger.info(f"Val set: {len(val_set)} tasks")
        logger.info(f"Test set: {len(test_set)} tasks")

        # Log dataset statistics
        mlflow.log_metric("train_size", len(train_set))
        mlflow.log_metric("val_size", len(val_set))
        mlflow.log_metric("test_size", len(test_set))
        mlflow.log_metric("train_positive_rate", train_set.period_labels.mean())  # type: ignore
        mlflow.log_metric("val_positive_rate", val_set.period_labels.mean())  # type: ignore
        mlflow.log_metric("test_positive_rate", test_set.period_labels.mean())  # type: ignore

        # Create data loaders
        train_loader = LabelledCriteriaTaskLoader(
            train_set, batch_size=actual_batch_size, device=device, seed=42
        )
        val_loader = LabelledCriteriaTaskLoader(
            val_set, batch_size=actual_batch_size, device=device, seed=0
        )
        test_loader = LabelledCriteriaTaskLoader(
            test_set, batch_size=actual_batch_size, device=device, seed=0
        )

        # Create base NP model
        logger.info("Creating base Neural Process model...")
        np_model = create_model(
            train_set.num_measurements,
            length_scale,
            channels=channels,
            num_layers=num_layers,
            margin=margin,
        )
        np_model.load_state_dict(model_data["weights"])
        np_model.to(device)

        # Create classifier
        logger.info("Creating classification head...")
        classifier = NPClassifier(
            np_model=np_model,
            hidden_dims=hidden_dims_list,
            pooling=pooling,
            dropout=dropout,
            freeze_np=freeze_np,
            num_classes=2,
        ).to(device)

        logger.info("Classification head architecture:")
        logger.info(f"  Input dim: {classifier.input_dim}")
        logger.info(f"  Hidden dims: {hidden_dims_list}")
        logger.info(f"  Pooling: {pooling}")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Freeze NP: {freeze_np}")

        # Create initial state
        state = B.create_random_state(torch.float32, seed=42)

        # Create objective function
        objective = partial(classification_objective)

        # Train the classification head
        logger.info("Starting training...")
        classifier, state = fit_model(
            state=state,
            model=classifier,
            train_batches=train_loader,
            eval_batches=val_loader,
            objective=objective,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            epsilon=1e-8,
            use_lr_scheduler=use_lr_scheduler,
        )

        logger.info("Training completed!")
        # load weights of best model
        model_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model-best.torch")
        model_data = torch.load(model_path, map_location=device)
        classifier.load_state_dict(model_data["weights"])
        classifier.to(device)
        classifier.eval()

        # Evaluate on test set using existing infrastructure
        logger.info("Collecting classification probabilities on test set...")
        from primitivo_model.nps.classification import predict_classification_probabilities
        from primitivo_model.nps.criteria import get_criteria_results_df

        test_probs = predict_classification_probabilities(classifier, test_loader)

        # Create results dataframe using existing function
        test_results = get_criteria_results_df(test_probs, test_set)

        # Log results
        log_dataframe_to_mlflow(test_results, "test_classification_preds.csv", run_id)

        # Use existing evaluation function for probability-based metrics
        test_metrics = evaluate_criteria_prediction(
            results_df=test_results,
            test_set=test_set,
            model_name="NP Classifier",
            plot_prefix="classifier",
            log_filename="test_classification_metrics.csv",
            run_id=run_id,
        )

        logger.info("Finding optimal threshold on validation set...")
        val_probs = predict_classification_probabilities(classifier, val_loader)
        val_results = get_criteria_results_df(val_probs, val_set)
        optimal_threshold = find_optimal_threshold(
            np.array(val_results["meets_criteria"]),
            np.array(val_results["prob_meets_criteria"]),
        )
        mlflow.log_metric("optimal_threshold", optimal_threshold)

        logger.info(f"✅ Training complete! Run ID: {run_id}")

        return run_id
