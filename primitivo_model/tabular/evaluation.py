"""
Evaluation utilities for tabular criteria prediction models.

This module provides evaluation functions and metrics that match
the neural process evaluation framework.
"""

from collections import defaultdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from primitivo_model.evaluation.criteria import compute_precision_at_k


def extract_measurement_category_from_one_hot(
    features_df: pd.DataFrame, measurement_names: list[str]
) -> np.ndarray:
    """
    Extract measurement category indices from one-hot encoded columns.

    Args:
        features_df: DataFrame containing one-hot encoded measurement columns (is_*)
        measurement_names: List of measurement names in the correct order

    Returns:
        Array of measurement category indices
    """
    is_columns = [f"is_{name}" for name in measurement_names]
    return features_df[is_columns].values.argmax(axis=1)


def evaluate_tabular_classification_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, test_set, model_name: str = "Tabular Model"
) -> Dict[str, float]:
    """
    Comprehensive evaluation of tabular criteria prediction model.

    Args:
        model: Fitted TabularCriteriaModel
        X_test: Test features
        y_test: Test labels
        test_set: DailyCriteriaTaskSet for computing precision@k
        model_name: Name for logging and plots

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model_name} on {len(X_test)} test samples")

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Calculate core metrics matching neural process evaluation
    metrics = {
        "auroc": roc_auc_score(y_test, y_pred_proba),
        "average_precision": average_precision_score(y_test, y_pred_proba),
        "brier_score": brier_score_loss(y_test, y_pred_proba),
    }

    # Calculate precision@5 using the same method as neural process
    # if test_set:
    try:
        results_df = pd.DataFrame(
            {"meets_criteria": y_test, "prob_meets_criteria": y_pred_proba}, index=y_test.index
        )

        precision_at_5 = compute_precision_at_k(test_set, results_df, k=5)
        metrics["precision_at_5"] = precision_at_5
    except Exception as e:
        logger.warning(f"Could not compute precision@5: {e}")
        metrics["precision_at_5"] = None

    # Log results
    logger.info(f"{model_name} Evaluation Results:")
    logger.info(f"  AUROC: {metrics['auroc']:.3f}")
    logger.info(f"  Average Precision: {metrics['average_precision']:.3f}")
    logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
    if metrics["precision_at_5"] is not None:
        logger.info(f"  Precision@5: {metrics['precision_at_5']:.3f}")

    return metrics


def evaluate_tabular_regression_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Tabular Regression Model"
) -> Dict[str, float]:
    """
    Comprehensive evaluation of tabular regression model.

    Args:
        model: Fitted TabularRegressionModel
        X_test: Test features
        y_test: Test labels
        model_name: Name for logging and plots

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model_name} on {len(X_test)} test samples")

    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate core regression metrics
    metrics = {
        "mae": float(np.mean(np.abs(y_pred - y_test))),
        "rmse": float(np.sqrt(np.mean((y_pred - y_test) ** 2))),
        "r2": float(model.model.score(X_test, y_test)),
    }

    # Log results
    logger.info(f"{model_name} Evaluation Results:")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")

    return metrics


def create_feature_importance_plot(
    model, top_k: int = 20, figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Create feature importance plot.

    Args:
        model: Fitted TabularCriteriaModel
        top_k: Number of top features to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    importance = model.get_feature_importance()

    if importance is None:
        raise ValueError("Model does not support feature importance")

    # Get top features
    top_features = importance.head(top_k)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(range(len(top_features)), top_features.values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_k} Feature Importances - {model.model_type.upper()}")

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_features.values)):
        ax.text(
            value + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    return fig


def collect_task_regression_predictions(
    task_pred_values,
    task_meas_categories,
    num_measurements,
):
    preds_by_cat = defaultdict(list)
    for cat, pred in zip(task_meas_categories, task_pred_values):
        preds_by_cat[cat].append(pred)

    # 2. Build the final list in the correct order
    task_predictions = []

    # Create the empty array template just once
    empty_array = np.empty((1, 1, 0))

    for meas_idx in range(num_measurements):
        if meas_idx in preds_by_cat:
            # Convert the list of preds to the required array format
            # We use np.array() which is fast.
            pred_array = np.array(preds_by_cat[meas_idx]).reshape(1, 1, -1)
            task_predictions.append(pred_array)
        else:
            # No predictions for this measurement
            task_predictions.append(empty_array)

    return task_predictions


def collect_regression_predictions(model, task_set, features_df: pd.DataFrame):
    """
    Optimized: Collects predictions for all tasks using groupby.

    Replaces the slow Python loop over tasks and full-dataset masking
    with a single, efficient pandas groupby operation.
    """

    all_predictions = model.predict(features_df)

    # Extract measurement category from one-hot encoded columns
    measurement_categories = extract_measurement_category_from_one_hot(
        features_df, task_set.measurement_names
    )

    proc_df = pd.DataFrame(
        {
            "task_id": features_df.index.get_level_values(0),
            "measurement_category": measurement_categories,
            "prediction": all_predictions,
        }
    )

    valid_tasks_set = set(task_set.tasks)
    proc_df = proc_df[proc_df["task_id"].isin(valid_tasks_set)]

    grouped_by_task = proc_df.groupby("task_id")
    task_preds = {}

    for task_id, task_data in grouped_by_task:
        task_pred_values = task_data["prediction"].values
        task_meas_categories = task_data["measurement_category"].values

        task_preds[task_id] = collect_task_regression_predictions(
            task_pred_values, task_meas_categories, task_set.num_measurements
        )

    return task_preds
