"""
Tabular models for direct criteria prediction.

This module implements tabular machine learning models that directly predict
whether patients will meet clinical criteria, bypassing the two-step approach
of first forecasting measurements then applying criteria.
"""

from .evaluation import (
    collect_regression_predictions,
    evaluate_tabular_classification_model,
    evaluate_tabular_regression_model,
)
from .feature_engineering import RegressionTabularFeatureExtractor, TabularFeatureExtractor
from .models import TabularCriteriaModel, TabularRegressionModel

__all__ = [
    "TabularFeatureExtractor",
    "RegressionTabularFeatureExtractor",
    "TabularCriteriaModel",
    "TabularRegressionModel",
    "evaluate_tabular_classification_model",
    "evaluate_tabular_regression_model",
    "collect_regression_predictions",
]
