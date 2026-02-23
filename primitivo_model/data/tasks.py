from typing import Any, Optional, Set, Tuple

import pandas as pd

import lab.torch as B


class TaskSplitter:
    """Base class for measurement splitting strategies"""

    def split_measurement(
        self, task_df: pd.DataFrame, measurement_names: list[str], dtype: Any
    ) -> Tuple:
        """Split a measurement group into context and target sets

        Args:
            sub_group: DataFrame containing measurements of a specific type
            name: Name of the measurement
            dtype: Data type for tensors

        Returns:
            Tuple of (context_x, context_y, target_x, target_y)
        """
        raise NotImplementedError

    def handle_empty_measurement(self, dtype):
        """Handle case when measurement group is empty"""
        empty_array = B.cast(dtype, B.zeros(dtype, 1, 1, 0))
        return empty_array, empty_array, empty_array, empty_array


class ForecastSplitter(TaskSplitter):
    """Strategy that splits measurements based on time rather than random selection"""

    def __init__(
        self, sparse_only: bool, forecast_hours: float, sparse_measurements: Optional[Set] = None
    ):
        self.sparse_only = sparse_only
        self.sparse_measurements = sparse_measurements

        if self.sparse_only and not self.sparse_measurements:
            raise ValueError("sparse_measurements must be provided if sparse_only is True")
        self.forecast_hours = forecast_hours

    def split_task(
        self,
        task_df,
        measurement_names,
        prediction_time,
        dtype,
    ):
        enc_ctx = []
        enc_xt = []
        enc_yt = []

        # OPTIMIZATION: Pre-group data by measurement name to avoid repeated filtering
        grouped_measurements = {}
        if len(task_df) > 0:
            for name, group in task_df.groupby("name"):
                grouped_measurements[name] = group.sort_values("enc_elapsed_time")

        # Process each measurement type
        for i, name in enumerate(measurement_names):
            sub_group = grouped_measurements.get(name, None)

            if sub_group is None or len(sub_group) == 0:
                meas_x_ctx, meas_y_ctx, meas_x_trg, meas_y_trg = self.handle_empty_measurement(
                    dtype
                )
            else:
                # OPTIMIZATION: Vectorized time comparison
                times = sub_group.enc_elapsed_time.values
                values = sub_group.value.values
                history_mask = times < prediction_time

                meas_x = times[None, None, :]
                meas_y = values[None, None, :]

                meas_x_ctx = meas_x[:, :, history_mask]
                meas_y_ctx = meas_y[:, :, history_mask]
                meas_x_trg = meas_x[:, :, ~history_mask]
                meas_y_trg = meas_y[:, :, ~history_mask]

            # we only want to eval for the sparse measurements
            if self.sparse_only and name not in self.sparse_measurements:
                meas_x_trg = B.cast(dtype, B.zeros(1, 1, 0))
                meas_y_trg = B.cast(dtype, B.zeros(1, 1, 0))

            enc_ctx.append((meas_x_ctx, meas_y_ctx))
            enc_xt.append(meas_x_trg)
            enc_yt.append(meas_y_trg)

        return enc_ctx, enc_xt, enc_yt


class GriddedForecastSplitter(TaskSplitter):
    def __init__(
        self,
        forecast_grid,
    ):
        self.forecast_grid = forecast_grid
        self.forecast_hours = forecast_grid[-1] - forecast_grid[0]

    def split_task(
        self,
        task_df,
        measurement_names,
        prediction_time,
        dtype,
    ):
        enc_ctx = []
        enc_xt = []

        # OPTIMIZATION: Pre-group data by measurement name to avoid repeated filtering
        grouped_measurements = {}
        if len(task_df) > 0:
            for name, group in task_df.groupby("name"):
                grouped_measurements[name] = group.sort_values("enc_elapsed_time")

        # Process each measurement type
        for i, name in enumerate(measurement_names):
            sub_group = grouped_measurements.get(name, None)

            if sub_group is None or len(sub_group) == 0:
                meas_x_ctx, meas_y_ctx, meas_x_trg, _ = self.handle_empty_measurement(dtype)
            else:
                # OPTIMIZATION: Vectorized time comparison
                times = sub_group.enc_elapsed_time.values
                values = sub_group.value.values
                history_mask = times < prediction_time

                meas_x = times[None, None, :]
                meas_y = values[None, None, :]

                meas_x_ctx = meas_x[:, :, history_mask]
                meas_y_ctx = meas_y[:, :, history_mask]

                meas_x_trg = (prediction_time + self.forecast_grid)[None, None, :]

            enc_ctx.append((meas_x_ctx, meas_y_ctx))
            enc_xt.append(meas_x_trg)

        return enc_ctx, enc_xt, None
