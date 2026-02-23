"""
CLI commands for tabular criteria prediction models.

This module provides command-line interface for training and evaluating
tabular models that directly predict clinical criteria.
"""

import random
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import typer
from loguru import logger
from matplotlib import pyplot as plt

from primitivo_model.data.criteria import DailyCriteriaTaskSet
from primitivo_model.data.generator import SampledForecastTaskSet
from primitivo_model.evaluation.criteria import (
    evaluate_criteria_prediction,
    find_optimal_threshold,
)
from primitivo_model.mlflow_utils import (
    log_dataframe_to_mlflow,
    log_plot_to_mlflow,
    log_sklearn_model_to_mlflow,
    mlflow,
)
from primitivo_model.plots import (
    IVSwitchingAnalysisPlotter,
    make_and_log_task_plots,
    plot_task_tabular,
)
from primitivo_model.tabular import (
    TabularCriteriaModel,
    evaluate_tabular_classification_model,
    evaluate_tabular_regression_model,
)
from primitivo_model.nps.analysis import log_raw_mae_summary
from primitivo_model.tabular.evaluation import (
    collect_regression_predictions,
    create_feature_importance_plot,
)
from primitivo_model.tabular.feature_engineering import (
    CriteriaTabularFeatureExtractor,
    RegressionTabularFeatureExtractor,
)
from primitivo_model.tabular.models import TabularRegressionModel
from primitivo_model.util import print_banner

app = typer.Typer(help="Tabular models for criteria prediction")


# ============================================================================
# Helper Functions for Shared Abstractions
# ============================================================================


def parse_hyperparameter_list(param_str: str, param_type=float) -> list:
    """Parse comma-separated hyperparameter values."""
    return [param_type(x.strip()) for x in param_str.split(",")]


def get_metric_direction(metric_name: str, task_type: str) -> bool:
    """
    Determine if a metric should be maximized or minimized.

    Returns:
        True if higher is better, False if lower is better
    """
    if task_type == "classification":
        # brier_score: lower is better; auroc, average_precision: higher is better
        return metric_name != "brier_score"
    else:  # regression
        # mae, rmse: lower is better; r2: higher is better
        return metric_name == "r2"


def validate_selection_metric(metric_name: str, task_type: str):
    """Validate selection metric for the given task type."""
    if task_type == "classification":
        valid_metrics = ["auroc", "average_precision", "brier_score"]
    else:  # regression
        valid_metrics = ["mae", "rmse", "r2"]

    if metric_name not in valid_metrics:
        raise typer.BadParameter(
            f"Invalid selection_metric '{metric_name}' for {task_type}. "
            f"Must be one of: {valid_metrics}"
        )


def generate_param_combinations(model_type: str, hyperparameters: dict, task_type: str) -> list:
    """
    Generate all hyperparameter combinations for grid search.

    Args:
        model_type: 'gbdt', 'logistic', or 'linear'
        hyperparameters: Dict mapping param names to lists of values
        task_type: 'classification' or 'regression'

    Returns:
        List of tuples containing parameter combinations
    """
    if model_type == "gbdt":
        # Both classification and regression use n_estimators
        return list(
            product(
                hyperparameters["n_estimators"],
                hyperparameters["max_depth"],
                hyperparameters["learning_rate"],
            )
        )
    elif model_type == "logistic":
        return [(C,) for C in hyperparameters["C"]]
    else:  # linear
        return [()]  # No hyperparameters to search


def params_tuple_to_dict(model_type: str, params_tuple: tuple, task_type: str) -> dict:
    """Convert parameter tuple to dictionary based on model type."""
    if model_type == "gbdt":
        # Both classification and regression use n_estimators
        return {
            "n_estimators": params_tuple[0],
            "max_depth": params_tuple[1],
            "learning_rate": params_tuple[2],
        }
    elif model_type == "logistic":
        return {"C": params_tuple[0]}
    else:  # linear
        return {}


def extract_run_params(best_run_id: str, parent_run_id: str | None = None) -> dict:
    """
    Extract parameters from MLflow run, trying parent if child doesn't have them.

    Args:
        best_run_id: Run ID to extract params from
        parent_run_id: Optional parent run ID to fallback to

    Returns:
        Dictionary of parameters
    """
    run_info = mlflow.get_run(best_run_id)
    params = run_info.data.params

    if parent_run_id and "route" not in params:
        logger.info(
            f"Parameters not found in child run {best_run_id}, loading from parent {parent_run_id}"
        )
        parent_run_info = mlflow.get_run(parent_run_id)
        params = parent_run_info.data.params

    return params


# ============================================================================
# Unified Training Function
# ============================================================================


def _train_tabular_model_generic(
    task_type: str,  # 'classification' or 'regression'
    model_type: str,
    model_class: type,
    feature_extractor_class: type,
    dataset_factory,
    dataset_params: dict,
    training_params: dict,
    hyperparameters: dict,
) -> tuple[str | None, str, float]:
    """
    Generic training function for both classification and regression tasks.

    Args:
        task_type: 'classification' or 'regression'
        model_type: Type of model ('gbdt', 'logistic', 'linear')
        model_class: Model class to instantiate
        feature_extractor_class: Feature extractor class
        dataset_factory: Function to create dataset (takes subset as argument)
        dataset_params: Parameters for dataset creation
        training_params: Training parameters (experiment_name, random_state, etc.)
        hyperparameters: Dict mapping param names to lists of values

    Returns:
        Tuple of (best_run_id, parent_run_id, best_val_metric)
    """
    print_banner(
        f"## 🏥 {model_type.upper()} {task_type.title()} Training with Hyperparameter Search 📊"
    )

    # Validate selection metric
    selection_metric = training_params["selection_metric"]
    validate_selection_metric(selection_metric, task_type)
    metric_is_higher_better = get_metric_direction(selection_metric, task_type)

    # Generate parameter combinations
    param_combinations = generate_param_combinations(model_type, hyperparameters, task_type)

    # Set MLflow experiment
    mlflow.set_experiment(training_params["experiment_name"])

    # Start parent MLflow run
    with mlflow.start_run() as parent_run:
        parent_run_id = parent_run.info.run_id
        logger.info(f"Starting parent {task_type} training run: {parent_run_id}")

        # Log parent run parameters
        mlflow.log_params(
            {
                "model_type": model_type,
                "task_type": task_type,
                **dataset_params,
                **training_params,
            }
        )

        # Create datasets once for all runs
        logger.info("Creating datasets...")
        train_set = dataset_factory("train")
        val_set = dataset_factory("val")

        measurement_names = train_set.measurement_names

        # Create feature extractor
        feature_extractor = feature_extractor_class(
            measurement_names=measurement_names,
            include_temporal_features=training_params["include_temporal_features"],
            include_trend_features=training_params["include_trend_features"],
            include_missing_features=training_params["include_missing_features"],
        )

        # Extract features for all splits
        logger.info("Extracting features...")
        X_train, y_train = feature_extractor.extract_dataset(train_set)
        # don't need any more and it uses a lot of memory
        del train_set
        X_val, y_val = feature_extractor.extract_dataset(val_set)
        del val_set

        # Log dataset statistics
        mlflow.log_param("num_train_samples", len(X_train))
        mlflow.log_param("num_val_samples", len(X_val))
        mlflow.log_param("num_features", len(X_train.columns))

        if task_type == "classification":
            mlflow.log_param("train_positive_rate", y_train.mean())
            mlflow.log_param("val_positive_rate", y_val.mean())

        # Hyperparameter search
        best_val_metric = float("-inf") if metric_is_higher_better else float("inf")
        best_run_id = None

        logger.info(f"Running hyperparameter search over {len(param_combinations)} combinations")

        for params_tuple in param_combinations:
            # Start nested run for this parameter combination
            with mlflow.start_run(nested=True) as child_run:
                child_run_id = child_run.info.run_id

                # Build model_kwargs from params
                model_kwargs = params_tuple_to_dict(model_type, params_tuple, task_type)

                # Log hyperparameters for this run
                mlflow.log_params(
                    {
                        "model_type": model_type,
                        "task_type": task_type,
                        **dataset_params,
                        **training_params,
                        **model_kwargs,
                    }
                )

                # Create and train model (add cuda_device for GBDT models)
                if model_type == "gbdt":
                    model = model_class(
                        model_type=model_type,
                        random_state=training_params["random_state"],
                        cuda_device=training_params.get("cuda_device", -1),
                        **model_kwargs,
                    )
                else:
                    model = model_class(
                        model_type=model_type,
                        random_state=training_params["random_state"],
                        **model_kwargs,
                    )

                # Train model
                model.fit(X_train, y_train)

                # Evaluate on validation set
                if task_type == "classification":
                    val_metrics = evaluate_tabular_classification_model(
                        model, X_val, y_val, None, "Validation"
                    )

                    # Get the metric value for model selection
                    val_metric_value = val_metrics.get(selection_metric)
                    if val_metric_value is None:
                        logger.warning(
                            f"Selection metric '{selection_metric}' not available, skipping this model"
                        )
                        continue

                    # Log validation metrics
                    for metric, value in val_metrics.items():
                        if value is not None:
                            mlflow.log_metric(f"val_{metric}", value)

                else:  # regression
                    val_metrics = evaluate_tabular_regression_model(
                        model, X_val, y_val, "Validation"
                    )

                    # Get the metric value for model selection
                    val_metric_value = val_metrics.get(selection_metric)

                    # Log validation metrics
                    for metric, value in val_metrics.items():
                        if value is not None:
                            mlflow.log_metric(f"val_{metric}", value)

                # Save model for this run
                model_path = f"model_tabular_{model_type}.pkl"
                log_sklearn_model_to_mlflow(model, model_path, child_run.info.run_id)

                # Track best model based on selected metric
                is_better = (metric_is_higher_better and val_metric_value > best_val_metric) or (
                    not metric_is_higher_better and val_metric_value < best_val_metric
                )

                if is_better:
                    best_val_metric = val_metric_value
                    best_run_id = child_run_id
                    logger.info(f"New best validation {selection_metric}: {val_metric_value:.4f}")

        # Log best run info to parent run
        if best_run_id is not None:
            mlflow.log_param("best_run_id", best_run_id)
            mlflow.log_metric(f"best_val_{selection_metric}", best_val_metric)

            logger.info("Hyperparameter search completed!")
            logger.info(f"Best validation {selection_metric}: {best_val_metric:.4f}")
            logger.info(f"Best run ID: {best_run_id}")
        else:
            logger.error("No best model found!")

    return best_run_id, parent_run_id, best_val_metric


def _train_tabular_model(
    model_type: str,
    route: str,
    data_source: str,
    smoke_test: bool,
    day_start_hour: int,
    forecast_hours: int,
    forecast_grid_size: int,
    lookback_hours: int,
    min_task_measurements: int,
    experiment_name: str,
    random_state: int,
    include_temporal_features: bool,
    include_trend_features: bool,
    include_missing_features: bool,
    selection_metric: str,
    refresh_cache: bool,
    hyperparameters: dict,
    tight_criteria: bool,
    use_gpu: bool = False,
    cuda_device: int = -1,
):
    """
    Wrapper for training classification models (criteria prediction).
    Delegates to generic training function.
    """

    def create_dataset(subset):
        return DailyCriteriaTaskSet(
            route=route,
            data_source=data_source,
            min_task_measurements=min_task_measurements,
            smoke_test=smoke_test,
            forecast_hours=forecast_hours,
            forecast_grid_size=forecast_grid_size,
            lookback_hours=lookback_hours,
            day_start_hour=day_start_hour,
            subset=subset,
            refresh_cache=refresh_cache,
            tight_criteria=tight_criteria,
        )

    dataset_params = {
        "route": route,
        "data_source": data_source,
        "smoke_test": smoke_test,
        "day_start_hour": day_start_hour,
        "forecast_hours": forecast_hours,
        "forecast_grid_size": forecast_grid_size,
        "lookback_hours": lookback_hours,
        "min_task_measurements": min_task_measurements,
        "refresh_cache": refresh_cache,
        "tight_criteria": tight_criteria,
    }

    training_params = {
        "experiment_name": experiment_name,
        "random_state": random_state,
        "include_temporal_features": include_temporal_features,
        "include_trend_features": include_trend_features,
        "include_missing_features": include_missing_features,
        "selection_metric": selection_metric,
        "use_gpu": use_gpu,
        "cuda_device": cuda_device,
    }

    return _train_tabular_model_generic(
        task_type="classification",
        model_type=model_type,
        model_class=TabularCriteriaModel,
        feature_extractor_class=CriteriaTabularFeatureExtractor,
        dataset_factory=create_dataset,
        dataset_params=dataset_params,
        training_params=training_params,
        hyperparameters=hyperparameters,
    )


@app.command(
    name="train-gbdt",
    short_help="Train GBDT model with hyperparameter search for criteria prediction",
)
def train_gbdt(
    cuda_device: int = typer.Option(-1, help="CUDA device ID to use for GPU training (-1 for CPU)"),
    route: str = typer.Option("simple-charts-dev", help="Route to use for the data"),
    data_source: str = typer.Option("mimic4", help="Data source to use ('mimic4')"),
    smoke_test: bool = typer.Option(
        False,
        help="Run a smoke test with a small dataset for quick validation",
    ),
    day_start_hour: int = typer.Option(9, help="Hour of day when daily periods start (0-23)"),
    forecast_hours: float = typer.Option(12, help="Hours to forecast for each task"),
    forecast_grid_size: int = typer.Option(3, help="Number of grid points in forecast window"),
    lookback_hours: int = typer.Option(48, help="Number of hours to look back for each task"),
    min_task_measurements: int = typer.Option(10, help="Minimum number of measurements per task"),
    experiment_name: str = typer.Option(
        "tabular_criteria_prediction", help="MLflow experiment name"
    ),
    random_state: int = typer.Option(42, help="Random state for reproducibility"),
    # Feature engineering options
    include_temporal_features: bool = typer.Option(
        True, help="Include temporal features (time since last measurement, etc.)"
    ),
    include_trend_features: bool = typer.Option(
        True, help="Include trend features (slopes, changes)"
    ),
    include_missing_features: bool = typer.Option(True, help="Include missing data features"),
    # GBDT hyperparameter search ranges
    n_estimators_list: str = typer.Option(
        "100", help="Comma-separated list of n_estimators values"
    ),
    max_depth_list: str = typer.Option("6", help="Comma-separated list of max_depth values"),
    learning_rate_list: str = typer.Option(
        "0.1", help="Comma-separated list of learning_rate values"
    ),
    # Model selection metric
    selection_metric: str = typer.Option(
        "auroc", help="Metric to use for model selection (auroc, average_precision, or brier_score)"
    ),
    refresh_cache: bool = typer.Option(False, help="Refresh the task cache"),
    tight_criteria: bool = typer.Option(
        True, help="Use tight (True) or loose (False) clinical criteria bounds"
    ),
):
    """Train GBDT model with hyperparameter search for direct criteria prediction."""

    # Determine GPU usage from cuda_device parameter
    use_gpu = cuda_device >= 0
    if use_gpu:
        logger.info(f"Using CUDA device {cuda_device} for LightGBM GPU training")
    else:
        logger.info("Using CPU for LightGBM training")

    # Set smoke test defaults
    if smoke_test:
        n_estimators_list = "5"
        learning_rate_list = "0.1"
        max_depth_list = "3"

    # Parse hyperparameter lists using helper function
    hyperparameters = {
        "n_estimators": parse_hyperparameter_list(n_estimators_list, int),
        "max_depth": parse_hyperparameter_list(max_depth_list, int),
        "learning_rate": parse_hyperparameter_list(learning_rate_list, float),
    }

    # Run training - this returns IDs and allows training objects to be GC'd
    best_run_id, parent_run_id, best_val_metric = _train_tabular_model(
        model_type="gbdt",
        route=route,
        data_source=data_source,
        smoke_test=smoke_test,
        day_start_hour=day_start_hour,
        forecast_hours=forecast_hours,
        forecast_grid_size=forecast_grid_size,
        lookback_hours=lookback_hours,
        min_task_measurements=min_task_measurements,
        experiment_name=experiment_name,
        random_state=random_state,
        include_temporal_features=include_temporal_features,
        include_trend_features=include_trend_features,
        include_missing_features=include_missing_features,
        selection_metric=selection_metric,
        refresh_cache=refresh_cache,
        hyperparameters=hyperparameters,
        tight_criteria=tight_criteria,
        use_gpu=use_gpu,
        cuda_device=cuda_device,
    )

    # At this point, all training objects (datasets, feature extractors, models) have been GC'd
    # Now run evaluation with fresh objects
    if best_run_id is not None:
        test_auroc = evaluate(
            best_run_id=best_run_id,
            smoke_test=smoke_test,
            parent_run_id=parent_run_id,
            refresh_cache=refresh_cache,
        )
        logger.info(f"Test AUROC: {test_auroc:.4f} logged to parent run {parent_run_id}")
    else:
        logger.error("No best run found - skipping evaluation")

    return best_val_metric


@app.command(
    name="train-logistic",
    short_help="Train Logistic Regression model with hyperparameter search for criteria prediction",
)
def train_logistic(
    route: str = typer.Option("simple-charts-dev", help="Route to use for the data"),
    data_source: str = typer.Option("mimic4", help="Data source to use ('mimic4')"),
    smoke_test: bool = typer.Option(
        False,
        help="Run a smoke test with a small dataset for quick validation",
    ),
    day_start_hour: int = typer.Option(9, help="Hour of day when daily periods start (0-23)"),
    forecast_hours: float = typer.Option(12, help="Hours to forecast for each task"),
    forecast_grid_size: int = typer.Option(3, help="Number of grid points in forecast window"),
    lookback_hours: int = typer.Option(48, help="Number of hours to look back for each task"),
    min_task_measurements: int = typer.Option(10, help="Minimum number of measurements per task"),
    experiment_name: str = typer.Option(
        "tabular_criteria_prediction", help="MLflow experiment name"
    ),
    random_state: int = typer.Option(42, help="Random state for reproducibility"),
    # Feature engineering options
    include_temporal_features: bool = typer.Option(
        True, help="Include temporal features (time since last measurement, etc.)"
    ),
    include_trend_features: bool = typer.Option(
        True, help="Include trend features (slopes, changes)"
    ),
    include_missing_features: bool = typer.Option(True, help="Include missing data features"),
    # Logistic Regression hyperparameter search range
    C_list: str = typer.Option(
        "1.0", help="Comma-separated list of regularization strength (C) values"
    ),
    # Model selection metric
    selection_metric: str = typer.Option(
        "auroc", help="Metric to use for model selection (auroc, average_precision, or brier_score)"
    ),
    refresh_cache: bool = typer.Option(False, help="Refresh the task cache"),
    tight_criteria: bool = typer.Option(
        True, help="Use tight (True) or loose (False) clinical criteria bounds"
    ),
):
    """Train Logistic Regression model with hyperparameter search for direct criteria prediction."""

    # Set smoke test defaults
    if smoke_test:
        C_list = "1.0"

    # Parse hyperparameter list using helper function
    hyperparameters = {
        "C": parse_hyperparameter_list(C_list, float),
    }

    # Run training - this returns IDs and allows training objects to be GC'd
    best_run_id, parent_run_id, best_val_metric = _train_tabular_model(
        model_type="logistic",
        route=route,
        data_source=data_source,
        smoke_test=smoke_test,
        day_start_hour=day_start_hour,
        forecast_hours=forecast_hours,
        forecast_grid_size=forecast_grid_size,
        lookback_hours=lookback_hours,
        min_task_measurements=min_task_measurements,
        experiment_name=experiment_name,
        random_state=random_state,
        include_temporal_features=include_temporal_features,
        include_trend_features=include_trend_features,
        include_missing_features=include_missing_features,
        selection_metric=selection_metric,
        refresh_cache=refresh_cache,
        hyperparameters=hyperparameters,
        tight_criteria=tight_criteria,
    )

    # At this point, all training objects (datasets, feature extractors, models) have been GC'd
    # Now run evaluation with fresh objects
    if best_run_id is not None:
        test_auroc = evaluate(
            best_run_id=best_run_id,
            smoke_test=smoke_test,
            parent_run_id=parent_run_id,
            refresh_cache=refresh_cache,
        )
        logger.info(f"Test AUROC: {test_auroc:.4f} logged to parent run {parent_run_id}")
    else:
        logger.error("No best run found - skipping evaluation")

    return best_val_metric


@app.command(
    name="train-regression",
    short_help="Train regression model with hyperparameter search for measurement prediction",
)
def train_regression(
    cuda_device: int = typer.Option(-1, help="CUDA device ID to use for GPU training (-1 for CPU)"),
    route: str = typer.Option("simple-charts-dev", help="Route to use for the data"),
    data_source: str = typer.Option("mimic4", help="Data source to use ('mimic4')"),
    smoke_test: bool = typer.Option(
        False,
        help="Run a smoke test with a small dataset for quick validation",
    ),
    forecast_hours: float = typer.Option(12, help="Hours to forecast for each task"),
    lookback_hours: int = typer.Option(48, help="Number of hours to look back for each task"),
    min_task_measurements: int = typer.Option(10, help="Minimum number of measurements per task"),
    experiment_name: str = typer.Option("tabular_regression", help="MLflow experiment name"),
    random_state: int = typer.Option(42, help="Random state for reproducibility"),
    # Feature engineering options
    include_temporal_features: bool = typer.Option(
        True, help="Include temporal features (time since last measurement, etc.)"
    ),
    include_trend_features: bool = typer.Option(
        True, help="Include trend features (slopes, changes)"
    ),
    include_missing_features: bool = typer.Option(True, help="Include missing data features"),
    # Model options
    model_type: str = typer.Option(
        "gbdt", help="Model type: 'gbdt' (gradient boosting) or 'linear'"
    ),
    # GBDT hyperparameter search ranges
    n_estimators_list: str = typer.Option(
        "100", help="Comma-separated list of n_estimators values (GBDT only)"
    ),
    max_depth_list: str = typer.Option(
        "6", help="Comma-separated list of max_depth values (GBDT only)"
    ),
    learning_rate_list: str = typer.Option(
        "0.1", help="Comma-separated list of learning_rate values (GBDT only)"
    ),
    # Model selection metric
    selection_metric: str = typer.Option(
        "mae", help="Metric to use for model selection (mae or rmse)"
    ),
    # Data options
    refresh_cache: bool = typer.Option(False, help="Refresh cached task sets"),
):
    """
    Train a tabular regression model with hyperparameter search for measurement prediction.

    This serves as a baseline for neural process regression models.
    Uses features from historical measurements to predict future measurement values.
    """
    print_banner(f"## 📊 {model_type.upper()} Regression Training with Hyperparameter Search 🏥")

    # Determine GPU usage from cuda_device parameter
    use_gpu = cuda_device >= 0
    if use_gpu:
        logger.info(f"Using CUDA device {cuda_device} for LightGBM GPU training")
    else:
        logger.info("Using CPU for LightGBM training")

    # Set smoke test defaults
    if smoke_test:
        n_estimators_list = "5"
        learning_rate_list = "0.1"
        max_depth_list = "3"

    # Parse hyperparameter lists using helper function
    if model_type == "gbdt":
        hyperparameters = {
            "n_estimators": parse_hyperparameter_list(n_estimators_list, int),
            "max_depth": parse_hyperparameter_list(max_depth_list, int),
            "learning_rate": parse_hyperparameter_list(learning_rate_list, float),
        }
    else:
        hyperparameters = {}

    # Run training with hyperparameter search
    best_run_id, parent_run_id, best_val_metric = _train_regression_model(
        model_type=model_type,
        route=route,
        data_source=data_source,
        smoke_test=smoke_test,
        forecast_hours=forecast_hours,
        lookback_hours=lookback_hours,
        min_task_measurements=min_task_measurements,
        experiment_name=experiment_name,
        random_state=random_state,
        include_temporal_features=include_temporal_features,
        include_trend_features=include_trend_features,
        include_missing_features=include_missing_features,
        selection_metric=selection_metric,
        refresh_cache=refresh_cache,
        hyperparameters=hyperparameters,
        use_gpu=use_gpu,
        cuda_device=cuda_device,
    )

    # Run evaluation with fresh objects
    if best_run_id is not None:
        test_mae = evaluate_regression(
            best_run_id=best_run_id,
            smoke_test=smoke_test,
            parent_run_id=parent_run_id,
            refresh_cache=refresh_cache,
            num_tasks_to_plot=5,
        )
        logger.info(f"Test MAE: {test_mae:.4f} logged to parent run {parent_run_id}")
    else:
        logger.error("No best run found - skipping evaluation")

    return best_val_metric


def _train_regression_model(
    model_type: str,
    route: str,
    data_source: str,
    smoke_test: bool,
    forecast_hours: float,
    lookback_hours: int,
    min_task_measurements: int,
    experiment_name: str,
    random_state: int,
    include_temporal_features: bool,
    include_trend_features: bool,
    include_missing_features: bool,
    selection_metric: str,
    refresh_cache: bool,
    hyperparameters: dict,
    use_gpu: bool = False,
    cuda_device: int = -1,
):
    """
    Wrapper for training regression models (measurement prediction).
    Delegates to generic training function.
    """

    def create_dataset(subset):
        return SampledForecastTaskSet(
            route=route,
            data_source=data_source,
            min_task_measurements=min_task_measurements,
            smoke_test=smoke_test,
            forecast_hours=forecast_hours,
            lookback_hours=lookback_hours,
            subset=subset,
            refresh_cache=refresh_cache,
        )

    dataset_params = {
        "route": route,
        "data_source": data_source,
        "smoke_test": smoke_test,
        "forecast_hours": forecast_hours,
        "lookback_hours": lookback_hours,
        "min_task_measurements": min_task_measurements,
        "refresh_cache": refresh_cache,
    }

    training_params = {
        "experiment_name": experiment_name,
        "random_state": random_state,
        "include_temporal_features": include_temporal_features,
        "include_trend_features": include_trend_features,
        "include_missing_features": include_missing_features,
        "selection_metric": selection_metric,
        "use_gpu": use_gpu,
        "cuda_device": cuda_device,
    }

    return _train_tabular_model_generic(
        task_type="regression",
        model_type=model_type,
        model_class=TabularRegressionModel,
        feature_extractor_class=RegressionTabularFeatureExtractor,
        dataset_factory=create_dataset,
        dataset_params=dataset_params,
        training_params=training_params,
        hyperparameters=hyperparameters,
    )


@app.command(
    name="evaluate-regression", short_help="Evaluate a trained regression model on test data"
)
def evaluate_regression(
    best_run_id: str = typer.Argument(..., help="MLflow run ID of the best model to evaluate"),
    smoke_test: bool = typer.Option(
        False,
        help="Run a smoke test with a small dataset for quick validation",
    ),
    parent_run_id: str = typer.Option(None, help="Parent run ID to log test metrics to"),
    num_tasks_to_plot: int = typer.Option(10, help="Number of test tasks to plot predictions for"),
    refresh_cache: bool = typer.Option(False, help="Refresh cached task sets"),
):
    """
    Evaluate a trained tabular regression model on the test set.

    Loads the best model from training, runs predictions on test data,
    computes MAE analysis, and generates prediction plots.
    """
    print_banner("## 📊 Regression Model Evaluation 🔍")

    # Extract parameters using helper function
    orig_params = extract_run_params(best_run_id, parent_run_id)

    # Extract parameters from training run
    try:
        route = orig_params["route"]
        data_source = orig_params["data_source"]
        forecast_hours = float(orig_params["forecast_hours"])
        lookback_hours = float(orig_params["lookback_hours"])
        model_type = orig_params.get("model_type", "gbdt")
        min_task_measurements = int(orig_params["min_task_measurements"])
        include_temporal_features = orig_params["include_temporal_features"].lower() == "true"
        include_trend_features = orig_params["include_trend_features"].lower() == "true"
        include_missing_features = orig_params["include_missing_features"].lower() == "true"
    except KeyError as e:
        missing_param = str(e).strip("'")
        raise typer.BadParameter(f"Required parameter '{missing_param}' not found in original run")

    # If parent_run_id is provided, log to it. Otherwise log to the best_run_id
    context_run_id = parent_run_id if parent_run_id else best_run_id

    with mlflow.start_run(run_id=context_run_id) as run:
        logger.info(f"Loading model from run {best_run_id}")

        # Load the model
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        # Download the model artifact (returns path to the file itself, not a directory)
        model_path = client.download_artifacts(best_run_id, f"model_tabular_{model_type}.pkl")
        model = TabularRegressionModel.load_model(model_path)

        logger.info("Creating test dataset...")
        test_set = SampledForecastTaskSet(
            route=route,
            data_source=data_source,
            min_task_measurements=min_task_measurements,
            smoke_test=smoke_test,
            forecast_hours=forecast_hours,
            lookback_hours=lookback_hours,
            subset="test",
            refresh_cache=refresh_cache,
        )

        # Extract features
        logger.info("Extracting test features...")
        feature_extractor = RegressionTabularFeatureExtractor(
            measurement_names=test_set.measurement_names,
            include_temporal_features=include_temporal_features,
            include_trend_features=include_trend_features,
            include_missing_features=include_missing_features,
        )

        X_test, y_test = feature_extractor.extract_dataset(test_set)

        logger.info(f"Test samples: {len(X_test)}")
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_metric("test_tasks", len(test_set.tasks))

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_predictions = model.predict(X_test)
        test_mae = np.mean(np.abs(test_predictions - y_test))
        test_rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))
        test_r2 = model.model.score(X_test, y_test)

        logger.info(f"Test MAE: {test_mae:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.4f}")
        logger.info(f"Test R²: {test_r2:.4f}")

        mlflow.log_metric("test_mae", float(test_mae))
        mlflow.log_metric("test_rmse", float(test_rmse))
        mlflow.log_metric("test_r2", float(test_r2))

        test_task_preds = collect_regression_predictions(model, test_set, X_test)
        log_raw_mae_summary(test_task_preds, test_set, context_run_id)

        # Feature importance
        if model.get_feature_importance() is not None:
            importance_fig = create_feature_importance_plot(model, top_k=20)
            log_plot_to_mlflow(importance_fig, "feature_importance.png")
            plt.close(importance_fig)

            # Log feature importance as CSV
            importance_df = model.get_feature_importance().to_frame()
            log_dataframe_to_mlflow(importance_df, "feature_importance.csv", context_run_id)

        # Generate prediction plots for selected tasks
        logger.info(f"Generating prediction plots for {num_tasks_to_plot} tasks...")

        plot_task_fn = partial(
            plot_task_tabular, model, feature_extractor, hours_back=lookback_hours
        )
        # Generate and log prediction plots
        make_and_log_task_plots(
            plot_task_fn, num_tasks_to_plot if not smoke_test else 1, test_set, run.info.run_id
        )

        logger.info("Evaluation completed!")
        logger.info(f"Test MAE: {test_mae:.4f}")

        return test_mae


@app.command(name="evaluate", short_help="Evaluate a trained tabular model on test data")
def evaluate(
    best_run_id: str = typer.Argument(
        ..., help="MLflow run ID of the best trained model to evaluate"
    ),
    smoke_test: bool = typer.Option(
        False,
        help="Run a smoke test with a small dataset for quick validation",
    ),
    parent_run_id: str = typer.Option(
        None, help="Parent run ID to log evaluation results to (if different from model run)"
    ),
    refresh_cache: bool = typer.Option(False, help="Refresh the task cache"),
):
    """Evaluate a trained tabular model on test data with comprehensive analysis."""
    print_banner("## 📉 Tabular Model Evaluation 📈")

    # Get run information from the model run
    run_info = mlflow.get_run(best_run_id)
    experiment_id = run_info.info.experiment_id

    orig_params = run_info.data.params

    # Extract required parameters from the original training run
    try:
        route = orig_params["route"]
        data_source = orig_params["data_source"]
        model_type = orig_params.get(
            "model_type", "gbdt"
        )  # Default to gbdt for backwards compatibility
        day_start_hour = int(orig_params.get("day_start_hour", 3))
        forecast_hours = float(orig_params["forecast_hours"])
        lookback_hours = float(orig_params["lookback_hours"])
        min_task_measurements = int(orig_params["min_task_measurements"])
        forecast_grid_size = int(
            orig_params.get("forecast_grid_size", 3)
        )  # Default to 3 for backwards compatibility
        include_temporal_features = orig_params["include_temporal_features"].lower() == "true"
        include_trend_features = orig_params["include_trend_features"].lower() == "true"
        include_missing_features = orig_params["include_missing_features"].lower() == "true"
        tight_criteria = orig_params["tight_criteria"].lower() == "true"
    except KeyError as e:
        missing_param = str(e).strip("'")
        raise typer.BadParameter(f"Required parameter '{missing_param}' not found in original run")

    # Determine which run to log results to (parent if provided, otherwise the model run)
    target_run_id = parent_run_id if parent_run_id is not None else best_run_id

    # Start evaluation run context
    with mlflow.start_run(run_id=target_run_id, experiment_id=experiment_id):
        logger.info(f"Loading model from run {best_run_id} for evaluation")
        if parent_run_id:
            logger.info(f"Logging evaluation results to parent run {parent_run_id}")

        # Load the model from the specified best_run_id
        client = mlflow.tracking.MlflowClient()
        model_path = client.download_artifacts(best_run_id, f"model_tabular_{model_type}.pkl")
        model = TabularCriteriaModel.load_model(model_path)

        test_set = DailyCriteriaTaskSet(
            route=route,
            data_source=data_source,
            min_task_measurements=min_task_measurements,
            smoke_test=smoke_test,
            forecast_hours=forecast_hours,
            forecast_grid_size=forecast_grid_size,
            lookback_hours=lookback_hours,
            day_start_hour=day_start_hour,
            subset="test",
            refresh_cache=refresh_cache,
            tight_criteria=tight_criteria,
        )

        # Create feature extractor with same settings
        feature_extractor = CriteriaTabularFeatureExtractor(
            measurement_names=test_set.measurement_names,
            include_temporal_features=include_temporal_features,
            include_trend_features=include_trend_features,
            include_missing_features=include_missing_features,
        )

        # Extract test features
        X_test, y_test = feature_extractor.extract_dataset(test_set)
        mlflow.log_metric("num_test_samples", len(X_test))
        mlflow.log_metric("test_positive_rate", y_test.mean())

        logger.info("Running comprehensive test evaluation...")

        # Get predictions
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]

        # Prepare results dataframe for precision@5 calculation
        results_df = pd.DataFrame(
            {"meets_criteria": y_test, "prob_meets_criteria": y_test_pred_proba}, index=y_test.index
        )

        log_dataframe_to_mlflow(results_df, f"test_preds_{best_run_id}.csv", best_run_id)

        # Use unified evaluation function
        test_metrics = evaluate_criteria_prediction(
            results_df=results_df,
            test_set=test_set,
            model_name="Test",
            plot_prefix="test",
            log_filename="test_criteria_metrics.csv",
            run_id=best_run_id,
        )

        logger.info("Loading validation data to find optimal threshold...")
        val_set = DailyCriteriaTaskSet(
            route=route,
            data_source=data_source,
            min_task_measurements=min_task_measurements,
            smoke_test=smoke_test,
            forecast_hours=forecast_hours,
            forecast_grid_size=forecast_grid_size,
            lookback_hours=lookback_hours,
            day_start_hour=day_start_hour,
            subset="val",
            refresh_cache=refresh_cache,
            tight_criteria=tight_criteria,
        )
        X_val, y_val = feature_extractor.extract_dataset(val_set)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        optimal_threshold = find_optimal_threshold(
            np.array(y_val.values), np.array(y_val_pred_proba)
        )
        mlflow.log_metric("optimal_threshold", optimal_threshold)

        # Feature importance plot
        if model.get_feature_importance() is not None:
            importance_fig = create_feature_importance_plot(model, top_k=20)
            log_plot_to_mlflow(importance_fig, "feature_importance.png")

        # Generate IV switching analysis plots for selected patients
        logger.info("Generating IV switching analysis plots...")

        # Join results with period data and IV labels
        joined_periods_preds_labels = test_set.daily_periods_df.join(results_df, on="task_name")
        joined_periods_preds_labels = joined_periods_preds_labels.loc[
            joined_periods_preds_labels.meets_criteria.notna()
        ]

        # Select random patients for plotting (same number as would be plotted in other visualizations)
        unique_patients = joined_periods_preds_labels["pat_enc_csn_id"].unique()

        # Use same number as other plotting functions typically use
        num_patients_to_plot = min(15, len(unique_patients))
        selected_patients = random.sample(list(unique_patients), num_patients_to_plot)

        plotter = IVSwitchingAnalysisPlotter()

        for i, patient_id in enumerate(selected_patients):
            fig = plotter.plot_iv_switching_analysis(
                patient_id, joined_periods_preds_labels, test_set
            )
            log_plot_to_mlflow(fig, f"iv_switching_analysis_patient_{i + 1}_{patient_id}.png")
            plt.close(fig)  # Close to free memory

        logger.info("Evaluation completed!")
        logger.info(f"Test AUROC: {test_metrics['auroc']:.4f}")
        logger.info(f"Test Average Precision: {test_metrics['average_precision']:.4f}")

        return test_metrics["auroc"]


if __name__ == "__main__":
    app()
