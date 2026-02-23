import copy
from enum import Enum

import pandas as pd
import typer
from loguru import logger

import lab as B
from primitivo_model.data.criteria import (
    DailyCriteriaTaskSet,
    get_last_period_basline_preds,
)
from primitivo_model.data.generator import SampledForecastTaskSet
from primitivo_model.evaluation.criteria import (
    evaluate_criteria_prediction,
)
from primitivo_model.mlflow_utils import log_dataframe_to_mlflow, mlflow
from primitivo_model.naive.criteria import predict_meets_criteria
from primitivo_model.naive.model import (
    collect_predictions as naive_collect_predictions,
)
from primitivo_model.naive.model import eval_epoch as naive_eval_epoch
from primitivo_model.nps.analysis import log_raw_mae_summary
from primitivo_model.nps.criteria import get_criteria_results_df
from primitivo_model.plots import make_and_log_task_plots, plot_task_naive
from primitivo_model.util import print_banner

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command(name="repeat-last", short_help="Evaluate the repeat‐last baseline")
def run_repeat_last(
    num_tasks_to_plot: int = typer.Option(5, help="Number of test tasks to plot predictions for"),
    smoke_test: bool = typer.Option(
        False, help="Run a smoke test with a small dataset for quick validation"
    ),
    experiment_name: str = typer.Option("repeat_last", help="MLflow experiment name"),
    forecast_hours: float = typer.Option(12.0, help="Hours to forecast for each task (encounter)"),
    route: str = typer.Option("mock", help="Route to use for the test data"),
    data_source: str = typer.Option("radix", help="Data source to use ('radix' or 'mimic')"),
    forecast_sparse_only: bool = typer.Option(
        False,
        help="Whether to forecast only the sparse target points in validation",
    ),
    lookback_hours: float = typer.Option(
        48, help="Number of hours to look back for each task (encounter)"
    ),
    min_task_measurements: int = typer.Option(10, help="Minimum number of measurements per task"),
    refresh_cache: bool = typer.Option(False, help="Refresh the task cache"),
):
    print_banner("## 📉 Repeat‐Last Baseline 📈")
    B.set_random_seed(42)
    mlflow.set_experiment(experiment_name)
    if smoke_test:
        num_tasks_to_plot = 1

    with mlflow.start_run() as run:
        params = copy.copy(locals())
        del params["experiment_name"]
        mlflow.log_params(params)
        mlflow.log_param("model_type", "repeat_last")

        test_gen = SampledForecastTaskSet(
            forecast_hours=forecast_hours,
            lookback_hours=lookback_hours,
            subset="test",
            route=route,
            sparse_only=forecast_sparse_only,
            smoke_test=smoke_test,
            data_source=data_source,
            min_task_measurements=min_task_measurements,
            tasks_per_day=0.5,  # Fixed to 0.5 for test tasks
            refresh_cache=refresh_cache,
        )

        logger.info("Starting repeat‐last evaluation...")
        test_mae = naive_eval_epoch(test_gen)
        mlflow.log_metric("test_mae", test_mae.item())
        logger.info(f"Test loss (MAE): {test_mae:.4f}")

        task_preds = naive_collect_predictions(test_gen)
        log_raw_mae_summary(task_preds, test_gen, run.info.run_id)

        make_and_log_task_plots(
            plot_task_naive,
            num_tasks_to_plot,
            test_gen,
            run.info.run_id,
        )


class ModelType(str, Enum):
    dominant_class = "dominant_class"
    repeat_value = "repeat_value"
    repeat_label = "repeat_label"


@app.command(
    name="repeat-last-criteria",
    short_help="Evaluate the repeat‐last baseline on criteria prediction",
)
def evaluate_criteria(
    model_type: ModelType = typer.Option(
        ModelType.repeat_value,
        case_sensitive=False,
        help="Type of baseline model to use: dominant_class, repeat_value, or repeat_label",
    ),
    num_tasks_to_plot: int = typer.Option(5, help="Number of test tasks to plot predictions for"),
    smoke_test: bool = typer.Option(
        False, help="Run a smoke test with a small dataset for quick validation"
    ),
    experiment_name: str = typer.Option("repeat_last_criteria", help="MLflow experiment name"),
    forecast_hours: float = typer.Option(12.0, help="Hours to forecast for each task (encounter)"),
    forecast_grid_size: int = typer.Option(3, help="Number of grid points in forecast window"),
    route: str = typer.Option("mock", help="Route to use for the test data"),
    data_source: str = typer.Option("radix", help="Data source to use ('radix' or 'mimic')"),
    lookback_hours: float = typer.Option(
        48, help="Number of hours to look back for each task (encounter)"
    ),
    min_task_measurements: int = typer.Option(10, help="Minimum number of measurements per task"),
    day_start_hour: int = typer.Option(9, help="Hour of day when daily periods start (0-23)"),
    refresh_cache: bool = typer.Option(False, help="Refresh the task cache"),
    tight_criteria: bool = typer.Option(
        True, help="Use tight (True) or loose (False) clinical criteria bounds"
    ),
):
    print_banner("## 🏥 Repeat‐Last Criteria Evaluation 📊")
    B.set_random_seed(42)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        params = copy.copy(locals())
        del params["experiment_name"]
        mlflow.log_params(params)

        # Create criteria test dataset
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

        logger.info("Making repeat-last criteria predictions...")

        if model_type == ModelType.repeat_value:
            # Use the last available value as the prediction (repeat-last)
            task_meets_pred = predict_meets_criteria(test_set)

        elif model_type == ModelType.dominant_class:
            task_meets_pred = pd.Series(
                [test_set.period_labels.mean() > 0.5] * len(test_set.period_labels),
                index=test_set.period_labels.index,
            )

        elif model_type == ModelType.repeat_label:
            # Use the last available label as the prediction
            task_meets_pred = get_last_period_basline_preds(
                test_set.daily_periods_df, test_set.period_labels
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        # Get model predictions using the naive baseline

        model_results = get_criteria_results_df(task_meets_pred, test_set)

        log_dataframe_to_mlflow(
            model_results, f"test_criteria_preds_{run.info.run_id}.csv", run.info.run_id
        )

        # Use unified evaluation function
        test_metrics = evaluate_criteria_prediction(
            results_df=model_results,
            test_set=test_set,
            model_name="Test",
            plot_prefix="test",
            log_filename="test_criteria_metrics.csv",
            run_id=run.info.run_id,
        )

        return test_metrics["auroc"]


"""
Usage:
primitivo-model baseline evaluate \
    --num-tasks-to-plot 5 \
    --smoke-test 
"""
