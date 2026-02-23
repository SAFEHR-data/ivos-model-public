"""
Simple integration tests for tabular CLI commands.

These tests verify that the training functions work correctly
with minimal datasets.
"""

from primitivo_model.cli.tabular import (
    _train_regression_model,
    _train_tabular_model,
)


def test_train_gbdt_classification():
    """Test that GBDT classification training runs without error."""
    best_run_id, parent_run_id, best_val_metric = _train_tabular_model(
        model_type="gbdt",
        route="simple-charts-dev",
        data_source="mimic4",
        smoke_test=True,
        day_start_hour=9,
        forecast_hours=12,
        forecast_grid_size=3,
        lookback_hours=48,
        min_task_measurements=10,
        experiment_name="Default",
        random_state=42,
        include_temporal_features=True,
        include_trend_features=True,
        include_missing_features=True,
        selection_metric="auroc",
        refresh_cache=False,
        hyperparameters={
            "n_estimators": [5],
            "max_depth": [3],
            "learning_rate": [0.1],
        },
        tight_criteria=True,
    )
    
    assert best_run_id is not None, "No best run was selected"
    assert parent_run_id is not None, "No parent run was created"
    assert isinstance(best_val_metric, float), "Metric should be a float"
    assert best_val_metric > 0, "AUROC should be positive"


def test_train_gbdt_regression():
    """Test that GBDT regression training runs without error."""
    best_run_id, parent_run_id, best_val_metric = _train_regression_model(
        model_type="gbdt",
        route="simple-charts-dev",
        data_source="mimic4",
        smoke_test=True,
        forecast_hours=12,
        lookback_hours=48,
        min_task_measurements=10,
        experiment_name="Default",
        random_state=42,
        include_temporal_features=True,
        include_trend_features=True,
        include_missing_features=True,
        selection_metric="mae",
        refresh_cache=False,
        hyperparameters={
            "n_estimators": [5],
            "max_depth": [3],
            "learning_rate": [0.1],
        },
    )
    
    assert best_run_id is not None, "No best run was selected"
    assert parent_run_id is not None, "No parent run was created"
    assert isinstance(best_val_metric, float), "Metric should be a float"
    assert best_val_metric > 0, "MAE should be positive"


def test_train_linear_regression():
    """Test that linear regression training runs without error."""
    best_run_id, parent_run_id, best_val_metric = _train_regression_model(
        model_type="linear",
        route="simple-charts-dev",
        data_source="mimic4",
        smoke_test=True,
        forecast_hours=12,
        lookback_hours=48,
        min_task_measurements=10,
        experiment_name="Default",
        random_state=42,
        include_temporal_features=True,
        include_trend_features=True,
        include_missing_features=True,
        selection_metric="mae",
        refresh_cache=False,
        hyperparameters={},
    )
    
    assert best_run_id is not None, "No best run was selected"
    assert parent_run_id is not None, "No parent run was created"
    assert isinstance(best_val_metric, float), "Metric should be a float"
    assert best_val_metric > 0, "MAE should be positive"
