"""
Unified evaluation utilities for criteria prediction across all models.

This module provides common functions for evaluating classification performance
that are used across baseline, tabular, and neural process models.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from primitivo_model.mlflow_utils import log_dataframe_to_mlflow, mlflow


def compute_precision_at_k(test_set, results_df, k=1, min_patients_per_day=2):
    """
    Compute precision at k.

    Args:
        test_set: Test dataset
        results_df: DataFrame with meets_criteria and prob_meets_criteria columns
        k: Number of top predictions per day to evaluate
        min_patients_per_day: Minimum patients required per day for evaluation

    Returns:
        Precision@k score
    """
    joined_pred_periods = test_set.daily_periods_df.join(results_df, on="task_name").dropna()
    if test_set.data_source.name.startswith("mimic4"):
        # this is the reconstructed admission time for mimic
        adm_series = test_set.adm_df.set_index("pat_enc_csn_id").min_real_admittime.rename(
            "admittime"
        )
        joined_pred_periods = joined_pred_periods.drop(columns=["admittime"]).join(
            adm_series, on="pat_enc_csn_id"
        )

    # Add start_date column
    joined_pred_periods = joined_pred_periods.assign(
        start_date=(
            joined_pred_periods.admittime
            + joined_pred_periods.period_start.apply(pd.to_timedelta, unit="h")
        ).dt.date
    )

    # Group by start_date and log filtering step
    grouped = joined_pred_periods.groupby("start_date")
    rows_before_filter = len(joined_pred_periods)

    filtered_periods = grouped.filter(lambda x: len(x) >= min_patients_per_day)
    rows_after_filter = len(filtered_periods)
    rows_dropped = rows_before_filter - rows_after_filter

    logger.info(
        f"Filtered out {rows_dropped} rows ({rows_dropped / rows_before_filter:.1%}) with fewer than {min_patients_per_day} patients per day"
    )

    daily_precision_df = (
        filtered_periods.sample(
            frac=1
        )  # shuffle rows so order for non-prob predcitions will be random
        .sort_values("prob_meets_criteria")
        .groupby("start_date")
        .tail(k)
        .groupby("start_date")
        .meets_criteria.mean()
    )

    if daily_precision_df.empty:
        logger.warning(
            f"No days with >= {min_patients_per_day} patients found; skipping precision@{k} logging"
        )
        return float("nan")

    active_run = mlflow.active_run()
    log_dataframe_to_mlflow(
        daily_precision_df, f"daily_precisions_at_{k}.csv", active_run.info.run_id
    )
    return daily_precision_df.mean()


def compute_classification_metrics(
    results_df: pd.DataFrame,
    test_set=None,
) -> Dict[str, float]:
    """
    Compute standard classification metrics for criteria prediction.

    Args:
        results_df: DataFrame with meets_criteria and prob_meets_criteria columns
        test_set: Test dataset (needed for precision@5 calculation)

    Returns:
        Dictionary of classification metrics
    """
    y_true = np.array(results_df["meets_criteria"])
    y_pred_proba = np.array(results_df["prob_meets_criteria"])

    metrics = {
        "auroc": roc_auc_score(y_true, y_pred_proba),
        "average_precision": average_precision_score(y_true, y_pred_proba),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
        "total_samples": len(y_true),
        "class_balance": y_true.mean(),
    }

    # Calculate precision@5 if test_set is provided
    if test_set is not None:
        try:
            precision_at_5 = compute_precision_at_k(
                test_set, results_df, k=5, min_patients_per_day=10
            )
            metrics["precision_at_5"] = precision_at_5
        except Exception as e:
            logger.warning(f"Could not compute precision@5: {e}")
            metrics["precision_at_5"] = None
    else:
        metrics["precision_at_5"] = None

    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    logger.info(f"Optimal threshold: {optimal_threshold:.4f} (F1: {f1_scores[optimal_idx]:.4f})")
    return optimal_threshold


def log_classification_results(
    metrics: Dict[str, float],
    filename: str = "criteria_metrics.csv",
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create and log a DataFrame with classification metrics to MLflow.

    Args:
        metrics: Dictionary of metrics
        filename: Name for the CSV file
        run_id: MLflow run ID (if None, uses current active run)

    Returns:
        DataFrame with the metrics
    """
    # Create results table
    results_data = pd.Series(metrics).round(4)

    if run_id is None:
        active_run = mlflow.active_run()
        if active_run is not None:
            run_id = active_run.info.run_id
        else:
            raise ValueError("No active MLflow run and no run_id provided")

    log_dataframe_to_mlflow(results_data, filename, run_id)
    return results_data


def log_classification_summary(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Log a summary of classification results to the logger.

    Args:
        metrics: Dictionary of classification metrics
        model_name: Name of the model for logging
    """
    logger.info(f"{model_name} Criteria Evaluation Results:")
    logger.info(f"Total samples: {metrics['total_samples']}")
    logger.info(f"Class balance: {metrics['class_balance']:.1%}")
    logger.info(f"AUROC: {metrics['auroc']:.3f}")
    logger.info(f"Average Precision: {metrics['average_precision']:.3f}")
    logger.info(f"Brier Score: {metrics['brier_score']:.4f}")
    if metrics["precision_at_5"] is not None:
        logger.info(f"Precision@5: {metrics['precision_at_5']:.3f}")


def evaluate_criteria_prediction(
    results_df: pd.DataFrame,
    test_set=None,
    model_name: str = "Model",
    plot_prefix: str = "model",
    log_filename: str = "criteria_metrics.csv",
    run_id: Optional[str] = None,
) -> Dict[str, float]:
    """
    Complete evaluation pipeline for criteria prediction.

    This function computes metrics, creates plots, logs results to MLflow,
    and prints a summary - providing a unified interface for all models.

    Args:
        results_df: DataFrame with meets_criteria and prob_meets_criteria columns
        test_set: Test dataset (needed for precision@5 calculation)
        model_name: Name of the model for plots and logging
        plot_prefix: Prefix for plot filenames
        log_filename: Name for the metrics CSV file
        run_id: MLflow run ID (if None, uses current active run)

    Returns:
        Dictionary of classification metrics
    """
    # Compute metrics
    metrics = compute_classification_metrics(results_df, test_set)

    # Log metrics to MLflow
    mlflow.log_metrics({plot_prefix + "_" + k: v for k, v in metrics.items()})

    # Log results table
    log_classification_results(metrics, log_filename, run_id)

    # Log summary
    log_classification_summary(metrics, model_name)

    return metrics


