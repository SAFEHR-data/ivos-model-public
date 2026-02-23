"""
Tabular models for criteria prediction.

This module implements wrapper classes for LightGBM and scikit-learn models
to predict clinical criteria directly from tabular features.
"""

import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression, LogisticRegression


class BaseTabularModel(ABC):
    """Base class for tabular models with shared functionality."""

    def __init__(self, model_type: str, random_state: int = 42, **model_kwargs):
        """
        Initialize base tabular model.

        Args:
            model_type: Type of model
            random_state: Random state for reproducibility
            **model_kwargs: Additional arguments for the underlying model
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model_kwargs = model_kwargs
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        self._feature_importance = None  # Cache for feature importance

        self._create_model()

    @abstractmethod
    def _create_model(self):
        """Create the underlying scikit-learn model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _log_training_performance(self, X: pd.DataFrame, y: pd.Series):
        """Log training performance metrics. Must be implemented by subclasses."""
        pass

    def _compute_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """
        Compute and cache feature importance.

        For LightGBM models, uses built-in feature importance.
        For other models, extracts from model attributes.

        Args:
            X: Feature matrix used for training
            y: Target values used for training
        """
        if hasattr(self.model, "feature_importances_"):
            # Tree-based models (LightGBM) with built-in feature importance
            self._feature_importance = pd.Series(
                self.model.feature_importances_, index=self.feature_names, name="importance"
            ).sort_values(ascending=False)
        elif hasattr(self.model, "coef_"):
            # Linear models
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = coef[0]
            self._feature_importance = pd.Series(
                np.abs(coef), index=self.feature_names, name="importance"
            ).sort_values(ascending=False)
        else:
            logger.warning(f"Feature importance not available for {self.model_type}")
            self._feature_importance = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model to training data.

        Args:
            X: Feature matrix
            y: Target values
        """
        logger.info(
            f"Training {self.model_type} model on {len(X)} samples with {len(X.columns)} features"
        )

        # Store feature names
        self.feature_names = list(X.columns)

        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True

        # Log training performance
        self._log_training_performance(X, y)

        # Compute and store feature importance
        self._compute_feature_importance(X, y)

        # Log feature importance if available
        if hasattr(self, "_feature_importance") and self._feature_importance is not None:
            self._log_feature_importance()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict target values.

        Args:
            X: Feature matrix

        Returns:
            Array of predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Ensure feature order matches training
        if self.feature_names is not None:
            X = X[self.feature_names]

        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores.

        For tree-based models, returns feature importances.
        For linear models, returns absolute values of coefficients.
        For HistGradientBoosting models, returns permutation importance.

        Returns:
            Series of feature importance scores, or None if not available
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        # Return cached feature importance if available
        if self._feature_importance is not None:
            return self._feature_importance

        # If not cached, try to compute it
        if hasattr(self.model, "feature_importances_"):
            # Tree-based models (GBDT, RF)
            importance = pd.Series(
                self.model.feature_importances_, index=self.feature_names, name="importance"
            ).sort_values(ascending=False)
            return importance
        elif hasattr(self.model, "coef_"):
            # Linear models
            # Handle both 1D and 2D coefficient arrays
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = coef[0]
            importance = pd.Series(
                np.abs(coef), index=self.feature_names, name="importance"
            ).sort_values(ascending=False)
            return importance
        else:
            logger.warning(f"Feature importance not available for {self.model_type}")
            return None

    def _log_feature_importance(self, top_k: int = 10):
        """Log top feature importances."""
        importance = self.get_feature_importance()
        if importance is not None:
            importance_type = (
                "coefficients"
                if "linear" in self.model_type or "logistic" in self.model_type
                else "feature importances"
            )
            logger.info(f"Top {top_k} most important features by {importance_type}:")
            for feature, score in importance.head(top_k).items():
                logger.info(f"  {feature}: {score:.4f}")

    def save_model(self, filepath: str):
        """
        Save model to file.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "model_kwargs": self.model_kwargs,
            "random_state": self.random_state,
            "feature_importance": self._feature_importance,  # Save cached feature importance
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """
        Load model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded model instance
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Create new instance
        instance = cls(
            model_type=model_data["model_type"],
            random_state=model_data["random_state"],
            **model_data["model_kwargs"],
        )

        # Set loaded attributes
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.is_fitted = True
        # Load cached feature importance if available (for backwards compatibility)
        instance._feature_importance = model_data.get("feature_importance", None)

        logger.info(f"Model loaded from {filepath}")
        return instance

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is not None:
            return self.model.get_params()
        return {}

    def __str__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        n_features = len(self.feature_names) if self.feature_names else "unknown"
        class_name = self.__class__.__name__
        return f"{class_name}(type={self.model_type}, features={n_features}, status={status})"


class TabularCriteriaModel(BaseTabularModel):
    """Wrapper class for LightGBM and scikit-learn models to predict clinical criteria."""

    def __init__(
        self,
        model_type: str = "gbdt",
        random_state: int = 42,
        cuda_device: int = -1,
        **model_kwargs,
    ):
        """
        Initialize tabular model.

        Args:
            model_type: Type of model ('gbdt', 'logistic')
            random_state: Random state for reproducibility
            cuda_device: CUDA device ID to use (-1 for CPU, >=0 for GPU)
            **model_kwargs: Additional arguments for the underlying model
                For GBDT: n_estimators, max_depth, learning_rate
                For logistic: C (regularization strength)
        """
        self.cuda_device = cuda_device
        super().__init__(model_type, random_state, **model_kwargs)

    def _create_model(self):
        """Create the underlying model (LightGBM or scikit-learn)."""
        if self.model_type == "gbdt":
            default_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": self.random_state,
                "verbose": -1,
                "force_col_wise": True,  # Better for wide datasets
            }

            # Add GPU support if cuda_device >= 0
            if self.cuda_device >= 0:
                default_params["device"] = "gpu"
                default_params["gpu_device_id"] = self.cuda_device
                logger.info(f"Using GPU device {self.cuda_device} for LightGBM")

            default_params.update(self.model_kwargs)
            self.model = lgb.LGBMClassifier(**default_params)

        elif self.model_type == "logistic":
            default_params = {
                "C": 1.0,
                "max_iter": 1000,
                "random_state": self.random_state,
                "solver": "lbfgs",
            }
            default_params.update(self.model_kwargs)
            self.model = LogisticRegression(**default_params)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        logger.info(f"Created {self.model_type} model with parameters: {self.model.get_params()}")

    def _log_training_performance(self, X: pd.DataFrame, y: pd.Series):
        """Log training performance metrics."""
        train_score = self.model.score(X, y)
        logger.info(f"Training accuracy: {train_score:.4f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Ensure feature order matches training
        if self.feature_names is not None:
            X = X[self.feature_names]

        return self.model.predict_proba(X)


class TabularRegressionModel(BaseTabularModel):
    """Wrapper class for LightGBM and scikit-learn regression models to predict measurements."""

    def __init__(
        self,
        model_type: str = "gbdt",
        random_state: int = 42,
        cuda_device: int = -1,
        **model_kwargs,
    ):
        """
        Initialize tabular regression model.

        Args:
            model_type: Type of model ('gbdt', 'linear')
            random_state: Random state for reproducibility
            cuda_device: CUDA device ID to use (-1 for CPU, >=0 for GPU)
            **model_kwargs: Additional arguments for the underlying model
                For GBDT: n_estimators, max_depth, learning_rate
                For linear: fit_intercept
        """
        self.cuda_device = cuda_device
        super().__init__(model_type, random_state, **model_kwargs)

    def _create_model(self):
        """Create the underlying model (LightGBM or scikit-learn)."""
        if self.model_type == "gbdt":
            default_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": self.random_state,
                "objective": "mae",  # Mean absolute error objective
                "verbose": -1,
                "force_col_wise": True,  # Better for wide datasets
            }

            # Add GPU support if cuda_device >= 0
            if self.cuda_device >= 0:
                default_params["device"] = "gpu"
                default_params["gpu_device_id"] = self.cuda_device
                logger.info(f"Using GPU device {self.cuda_device} for LightGBM")

            default_params.update(self.model_kwargs)
            self.model = lgb.LGBMRegressor(**default_params)

        elif self.model_type == "linear":
            default_params = {
                "fit_intercept": True,
            }
            default_params.update(self.model_kwargs)
            self.model = LinearRegression(**default_params)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        logger.info(
            f"Created {self.model_type} regression model with parameters: {self.model.get_params()}"
        )

    def _log_training_performance(self, X: pd.DataFrame, y: pd.Series):
        """Log training performance metrics."""
        train_score = self.model.score(X, y)
        logger.info(f"Training R² score: {train_score:.4f}")
