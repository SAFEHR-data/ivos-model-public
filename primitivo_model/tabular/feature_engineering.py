"""
Feature engineering for tabular models.

This module converts irregular time series measurements from DailyCriteriaTaskSet
into tabular features suitable for scikit-learn models.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


class TabularFeatureExtractor:
    """Extract tabular features from irregular time series measurements."""

    def __init__(
        self,
        measurement_names: List[str],
        include_temporal_features: bool = True,
        include_trend_features: bool = True,
        include_missing_features: bool = True,
        percentiles: List[float] = [25, 75],
    ):
        """
        Initialize feature extractor.

        Args:
            measurement_names: List of measurement names to extract features for
            include_temporal_features: Whether to include temporal features
            include_trend_features: Whether to include trend features
            include_missing_features: Whether to include missing data features
            percentiles: Percentiles to compute for each measurement
        """
        self.measurement_names = measurement_names
        self.include_temporal_features = include_temporal_features
        self.include_trend_features = include_trend_features
        self.include_missing_features = include_missing_features
        self.percentiles = percentiles

    def extract_dataset(self, task_set) -> Tuple[pd.DataFrame, pd.Series]:
        raise NotImplementedError

    def extract_features_from_task(self, ctx_data, prediction_time: float) -> pd.Series:
        """
        Extract features from a single task.

        Args:
            task_data: Task data containing context measurements

        Returns:
            Series of extracted features
        """
        features = {}

        # Extract context data - this is the historical measurements before prediction time
        if not ctx_data:
            # Return empty features if no context data
            return pd.Series(self._get_empty_features())

        # Process each measurement type
        for i, measurement_name in enumerate(self.measurement_names):
            if i < len(ctx_data):
                meas_x, meas_y = ctx_data[i]

                # Convert tensors to numpy if needed
                if hasattr(meas_x, "numpy"):
                    times = meas_x.numpy().flatten()
                    values = meas_y.numpy().flatten()
                else:
                    times = np.array(meas_x).flatten()
                    values = np.array(meas_y).flatten()

                # Extract features for this measurement
                meas_features = self._extract_measurement_features(
                    times, values, measurement_name, prediction_time
                )
                features.update(meas_features)
            else:
                # No data for this measurement type
                meas_features = self._get_empty_measurement_features(measurement_name)
                features.update(meas_features)

        return pd.Series(features)

    def _extract_measurement_features(
        self,
        times: np.ndarray,
        values: np.ndarray,
        measurement_name: str,
        prediction_time: float,
    ) -> Dict[str, float]:
        """Extract features from a single measurement type."""
        features = {}
        prefix = f"{measurement_name}_"

        if len(values) == 0:
            return self._get_empty_measurement_features(measurement_name)

        # Statistical features
        features[f"{prefix}mean"] = np.mean(values)
        features[f"{prefix}std"] = np.std(values) if len(values) > 1 else 0.0
        features[f"{prefix}min"] = np.min(values)
        features[f"{prefix}max"] = np.max(values)
        features[f"{prefix}median"] = np.median(values)
        features[f"{prefix}count"] = len(values)

        # Percentile features
        for p in self.percentiles:
            features[f"{prefix}p{int(p)}"] = np.percentile(values, p)

        # Temporal features
        if self.include_temporal_features and len(times) > 0:
            features[f"{prefix}time_since_last"] = prediction_time - np.max(
                times
            )  # Most recent time (negative)
            features[f"{prefix}time_span"] = np.max(times) - np.min(times)
            features[f"{prefix}measurement_frequency"] = len(times) / max(
                features[f"{prefix}time_span"], 1e-6
            )

        # Trend features
        if self.include_trend_features and len(values) > 1:
            # Simple linear trend
            time_diffs = times[1:] - times[:-1]
            value_diffs = values[1:] - values[:-1]

            if len(time_diffs) > 0 and np.sum(np.abs(time_diffs)) > 0:
                slopes = value_diffs / (time_diffs + 1e-6)
                features[f"{prefix}trend_slope_mean"] = np.mean(slopes)
                features[f"{prefix}trend_slope_std"] = np.std(slopes)
            else:
                features[f"{prefix}trend_slope_mean"] = 0.0
                features[f"{prefix}trend_slope_std"] = 0.0

            # Direction of change
            features[f"{prefix}final_change"] = values[-1] - values[0]
            features[f"{prefix}positive_changes"] = np.sum(value_diffs > 0)
            features[f"{prefix}negative_changes"] = np.sum(value_diffs < 0)
        else:
            # No trend features for single measurement
            if self.include_trend_features:
                features[f"{prefix}trend_slope_mean"] = 0.0
                features[f"{prefix}trend_slope_std"] = 0.0
                features[f"{prefix}final_change"] = 0.0
                features[f"{prefix}positive_changes"] = 0
                features[f"{prefix}negative_changes"] = 0

        return features

    def _get_empty_measurement_features(self, measurement_name: str) -> Dict[str, float]:
        """Get features filled with default values for missing measurements."""
        features = {}
        prefix = f"{measurement_name}_"

        # Statistical features with default values
        features[f"{prefix}mean"] = 0.0
        features[f"{prefix}std"] = 0.0
        features[f"{prefix}min"] = 0.0
        features[f"{prefix}max"] = 0.0
        features[f"{prefix}median"] = 0.0
        features[f"{prefix}count"] = 0

        # Percentile features
        for p in self.percentiles:
            features[f"{prefix}p{int(p)}"] = 0.0

        # Temporal features
        if self.include_temporal_features:
            features[f"{prefix}time_since_last"] = (
                999.0  # Large value indicating no recent measurement
            )
            features[f"{prefix}time_span"] = 0.0
            features[f"{prefix}measurement_frequency"] = 0.0

        # Trend features
        if self.include_trend_features:
            features[f"{prefix}trend_slope_mean"] = 0.0
            features[f"{prefix}trend_slope_std"] = 0.0
            features[f"{prefix}final_change"] = 0.0
            features[f"{prefix}positive_changes"] = 0
            features[f"{prefix}negative_changes"] = 0

        return features

    def _get_empty_features(self) -> Dict[str, float]:
        """Get completely empty feature set."""
        features = {}
        for measurement_name in self.measurement_names:
            features.update(self._get_empty_measurement_features(measurement_name))

        # Global missing features
        if self.include_missing_features:
            features["missing_measurement_types"] = len(self.measurement_names)
            features["measurement_completeness"] = 0.0

        return features

    def _add_global_missing_features(self, features_df: pd.DataFrame):
        """Add global features about missing measurements."""
        # Count total measurements per task
        count_cols = [col for col in features_df.columns if col.endswith("_count")]

        # Count missing measurement types
        features_df["missing_measurement_types"] = (features_df[count_cols] == 0).sum(axis=1)

        # Measurement completeness ratio
        features_df["measurement_completeness"] = (
            len(self.measurement_names) - features_df["missing_measurement_types"]
        ) / len(self.measurement_names)

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be generated."""
        feature_names = []

        for measurement_name in self.measurement_names:
            prefix = f"{measurement_name}_"

            # Statistical features
            feature_names.extend(
                [
                    f"{prefix}mean",
                    f"{prefix}std",
                    f"{prefix}min",
                    f"{prefix}max",
                    f"{prefix}median",
                    f"{prefix}count",
                ]
            )

            # Percentile features
            for p in self.percentiles:
                feature_names.append(f"{prefix}p{int(p)}")

            # Temporal features
            if self.include_temporal_features:
                feature_names.extend(
                    [
                        f"{prefix}time_since_last",
                        f"{prefix}time_span",
                        f"{prefix}measurement_frequency",
                    ]
                )

            # Trend features
            if self.include_trend_features:
                feature_names.extend(
                    [
                        f"{prefix}trend_slope_mean",
                        f"{prefix}trend_slope_std",
                        f"{prefix}final_change",
                        f"{prefix}positive_changes",
                        f"{prefix}negative_changes",
                    ]
                )

        # Global missing features
        if self.include_missing_features:
            feature_names.extend(["missing_measurement_types", "measurement_completeness"])

        return feature_names


class CriteriaTabularFeatureExtractor(TabularFeatureExtractor):
    def extract_dataset(self, task_set) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"Extracting features from {len(task_set.tasks)} tasks")

        all_features = []
        all_labels = []
        task_ids = []

        prediction_times = task_set.daily_periods_df.set_index("task_name").period_start.to_dict()
        for task_id, task_data in tqdm(task_set.tasks.items(), desc="Extracting tabular features"):
            # Extract features
            prediction_time = prediction_times[task_id]
            ctx_data = task_data.get("contexts", [])
            features = self.extract_features_from_task(ctx_data, prediction_time)
            all_features.append(features)

            # Get label
            label = task_set.period_labels.get(task_id, None)
            all_labels.append(label)
            task_ids.append(task_id)

        # Create DataFrame
        features_df = pd.DataFrame(all_features, index=task_ids)
        labels_series = pd.Series(all_labels, index=task_ids, name="meets_criteria")

        # Remove rows with missing labels
        valid_mask = ~labels_series.isna()
        features_df = features_df[valid_mask]
        labels_series = labels_series[valid_mask]

        # Add global missing features
        if self.include_missing_features:
            self._add_global_missing_features(features_df)

        logger.info(
            f"Extracted {len(features_df)} valid samples with {len(features_df.columns)} features"
        )
        logger.info(f"Label distribution: {labels_series.value_counts().to_dict()}")

        return features_df, labels_series.astype(bool)


class RegressionTabularFeatureExtractor(TabularFeatureExtractor):
    def get_task_features(self, contexts, xt, prediction_time):
        """
        Extract features as DataFrame (used for plotting and single-task operations).

        This is a convenience wrapper around get_task_features_as_array that converts
        the result to a DataFrame.
        """
        features_array, shared_feature_names = self.get_task_features_as_array(
            contexts, xt, prediction_time
        )
        column_names = self._get_column_names(shared_feature_names)
        return pd.DataFrame(features_array, columns=column_names)

    def get_task_features_as_array(self, contexts, xt, prediction_time):
        """Extract features as numpy array instead of DataFrame for memory efficiency."""
        shared_features = self.extract_features_from_task(contexts, prediction_time)

        time_feature = np.concat(xt, axis=2).squeeze((0, 1)) - prediction_time

        num_targets = len(time_feature)
        # Tile shared features - keep as numpy array
        all_shared_features = np.tile(shared_features.to_numpy(), (num_targets, 1))

        # Create one-hot encoding for each measurement type as arrays
        one_hot_arrays = []
        for i, measurement_name in enumerate(self.measurement_names):
            one_hot_feature = np.concat(
                [np.ones_like(xi) if j == i else np.zeros_like(xi) for j, xi in enumerate(xt)],
                axis=2,
            ).squeeze((0, 1))
            one_hot_arrays.append(one_hot_feature.astype(int))

        # Stack all features: shared + time + one-hot encodings
        time_feature_2d = time_feature.reshape(-1, 1)
        one_hot_stacked = (
            np.column_stack(one_hot_arrays) if one_hot_arrays else np.empty((num_targets, 0))
        )

        features_array = np.column_stack([all_shared_features, time_feature_2d, one_hot_stacked])

        # Add missing features if needed
        if self.include_missing_features:
            # Count missing measurement types from shared features
            count_cols_mask = np.array([col.endswith("_count") for col in shared_features.index])
            count_values = shared_features.to_numpy()[count_cols_mask]
            missing_count = np.sum(count_values == 0)
            completeness = (len(self.measurement_names) - missing_count) / len(
                self.measurement_names
            )

            # Tile these values for all targets
            missing_features = np.tile([missing_count, completeness], (num_targets, 1))
            features_array = np.column_stack([features_array, missing_features])

        return features_array, shared_features.index

    def get_task_targets_as_array(self, yt):
        """Extract targets as numpy array."""
        targets = np.concat(yt, axis=2).squeeze()
        if len(targets.shape) == 0:
            return np.array([])
        return targets

    def _get_column_names(self, shared_feature_names):
        """Generate column names for the feature array."""
        column_names = list(shared_feature_names)
        column_names.append("measurement_time")

        # Add one-hot encoding column names
        for measurement_name in self.measurement_names:
            column_names.append(f"is_{measurement_name}")

        # Add missing features if needed
        if self.include_missing_features:
            column_names.extend(["missing_measurement_types", "measurement_completeness"])

        return column_names

    def extract_dataset(self, task_set) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"Extracting features from {len(task_set.tasks)} tasks")

        all_features = []
        all_labels = []
        task_ids = []
        row_counts = []  # Track how many rows each task contributes

        prediction_times = task_set.task_windows_df.prediction_time
        shared_feature_names = None  # Will be set on first task

        for task_id, task_data in tqdm(task_set.tasks.items(), desc="Extracting tabular features"):
            # Extract features
            prediction_time = prediction_times[task_id]
            ctx_data = task_data["contexts"]
            xt = task_data["xt"]
            yt = task_data["yt"]

            features_array, feature_names = self.get_task_features_as_array(
                ctx_data, xt, prediction_time
            )
            targets_array = self.get_task_targets_as_array(yt)

            if len(targets_array) == 0:
                continue

            # Store column names from first valid task
            if shared_feature_names is None:
                shared_feature_names = feature_names

            all_features.append(features_array)
            all_labels.append(targets_array)
            task_ids.append(task_id)
            row_counts.append(len(targets_array))

        # Combine arrays efficiently using numpy
        features_array = np.vstack(all_features)
        labels_array = np.concatenate(all_labels)

        # Create multi-index for task IDs
        task_index = np.repeat(task_ids, row_counts)

        # Convert to DataFrame only once at the end
        column_names = self._get_column_names(shared_feature_names)
        features_df = pd.DataFrame(features_array, columns=column_names, index=task_index)
        labels_series = pd.Series(labels_array, index=task_index)

        logger.info(
            f"Extracted {len(features_df)} valid samples with {len(features_df.columns)} features"
        )

        return features_df, labels_series
