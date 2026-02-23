"""
Evaluation utilities for the primitivo_model package.
"""

from .criteria import (
    compute_classification_metrics,
    compute_precision_at_k,
    evaluate_criteria_prediction,
    find_optimal_threshold,
    log_classification_results,
    log_classification_summary,
)

__all__ = [
    "compute_classification_metrics",
    "compute_precision_at_k",
    "evaluate_criteria_prediction",
    "find_optimal_threshold",
    "log_classification_results",
    "log_classification_summary",
]
