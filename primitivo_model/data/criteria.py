import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

import lab as B
from primitivo_model.data.generator import TaskSet
from primitivo_model.data.sources import create_data_source
from primitivo_model.data.tasks import GriddedForecastSplitter


def get_period_start(timestamp, day_start_hour):
    """Get the start of the 24-hour period containing this timestamp"""
    if timestamp.hour >= day_start_hour:
        # Same day period (e.g., 8am-8am next day)
        period_start = timestamp.replace(hour=day_start_hour, minute=0, second=0, microsecond=0)
    else:
        # Previous day period
        period_start = (timestamp - pd.Timedelta(days=1)).replace(
            hour=day_start_hour, minute=0, second=0, microsecond=0
        )
    return period_start


def get_daily_periods_from_adm(
    adm_df: pd.DataFrame,
    abx_rx_df: pd.DataFrame,
    forecast_hours=12.0,
    lookback_hours=48.0,
    interval_hours=24.0,
    day_start_hour=9,
    max_hours_since_iv=36,
) -> pd.DataFrame:
    # Convert admission and encounter end times to period boundaries
    adm_df["admit_period_start"] = adm_df["admittime"].apply(
        lambda x: get_period_start(x, day_start_hour)
    )
    adm_df["admit_period_end"] = adm_df["dischtime"].apply(
        lambda x: get_period_start(x, day_start_hour)
    )

    # Calculate number of periods for each encounter
    adm_df["enc_num_days"] = (
        (adm_df["admit_period_end"] - adm_df["admit_period_start"]).dt.total_seconds() / (24 * 3600)
    ).astype(int) + 1

    # Create period sequences using explode
    adm_df["day_of_enc"] = adm_df["enc_num_days"].apply(lambda x: list(range(x)))
    encounter_periods = adm_df.explode("day_of_enc")

    # Calculate actual period start/end times
    encounter_periods["period_start"] = encounter_periods["admit_period_start"] + pd.to_timedelta(
        encounter_periods["day_of_enc"] * interval_hours, unit="hours"
    )
    encounter_periods["period_end"] = encounter_periods["period_start"] + pd.Timedelta(
        hours=forecast_hours
    )

    encounter_periods["task_name"] = (
        encounter_periods.pat_enc_csn_id.astype(str)
        + "n"
        + encounter_periods.day_of_enc.astype(str)
    )

    encounter_periods = encounter_periods[
        ["pat_enc_csn_id", "admittime", "period_start", "period_end", "task_name"]
    ]
    encounter_periods["period_start"] = (
        encounter_periods["period_start"] - encounter_periods["admittime"]
    ).dt.total_seconds() / 60**2
    encounter_periods["period_end"] = (
        encounter_periods["period_end"] - encounter_periods["admittime"]
    ).dt.total_seconds() / 60**2

    encs_pre = encounter_periods.pat_enc_csn_id.unique()
    encounter_periods = encounter_periods.loc[encounter_periods.period_start >= lookback_hours]
    encs_post = encounter_periods.pat_enc_csn_id.unique()
    logger.info(f"dropped {len(encs_pre) - len(encs_post)} encs shorter than lookback")

    # now we add on IV labels
    merged_abx_periods = pd.merge(encounter_periods, abx_rx_df, on="pat_enc_csn_id")
    merged_abx_periods = merged_abx_periods.loc[
        # current active prescription or prescription ended within 36 hrs
        (
            (merged_abx_periods.period_start >= merged_abx_periods.starttime)
            & (merged_abx_periods.period_start < (merged_abx_periods.stoptime + max_hours_since_iv))
        )
    ]

    merged_abx_periods["on_iv"] = merged_abx_periods["route"] == "IV"
    merged_abx_periods["on_po"] = merged_abx_periods["route"] != "IV"
    on_iv_label = merged_abx_periods.groupby("task_name").on_iv.any()
    on_po_label = merged_abx_periods.groupby("task_name").on_po.any()

    enc_first_iv_rx_time = (
        abx_rx_df[abx_rx_df.route == "IV"]
        .sort_values("starttime")
        .groupby("pat_enc_csn_id")
        .first()
        .starttime.rename("enc_first_iv_rx_time")
    )
    encounter_periods = encounter_periods.join(enc_first_iv_rx_time, on="pat_enc_csn_id")

    # Check if at least 24 hours have passed since first IV prescription
    iv_rx_enough_time_passed = (
        encounter_periods.set_index("task_name")["period_start"]
        - encounter_periods.set_index("task_name")["enc_first_iv_rx_time"]
    ) >= 24
    # Apply both conditions: patient must have active/recent IV AND at least 24hrs since first IV
    on_iv_label = (on_iv_label & iv_rx_enough_time_passed).rename("on_iv")
    on_iv_po_days_df = (
        (
            encounter_periods[["task_name"]]
            .join(on_iv_label, how="left", on="task_name")
            .join(on_po_label, how="left", on="task_name")
        )
        .set_index("task_name")
        .astype("boolean")
        .fillna(False)
    )

    daily_periods_df = encounter_periods.join(on_iv_po_days_df, on="task_name")
    daily_periods_df = daily_periods_df.loc[daily_periods_df.on_iv]

    return daily_periods_df


def get_period_criteria_labels(
    daily_periods_df, measurements_df, clinical_criteria, forecast_grid_size
):
    """
    Calculates period labels using a memory-efficient pd.merge_asof.
    """

    # 1. Prep criteria dicts (no change)
    lower_clinical_criteria = {k: v[0] for k, v in clinical_criteria.items()}
    upper_clinical_criteria = {k: v[1] for k, v in clinical_criteria.items()}

    meas_sorted = measurements_df.sort_values(by=["enc_elapsed_time"])
    periods_sorted = daily_periods_df.sort_values(by=["period_start"])

    merged_meas_periods = pd.merge_asof(
        meas_sorted,
        periods_sorted,
        left_on="enc_elapsed_time",
        right_on="period_start",
        by="pat_enc_csn_id",
        direction="backward",
    )

    merged_meas_periods = merged_meas_periods.loc[
        merged_meas_periods.enc_elapsed_time < merged_meas_periods.period_end
    ]

    relative_time = merged_meas_periods["enc_elapsed_time"] - merged_meas_periods.period_start
    # Divide relative_time by the bin_width before flooring
    bin_index = np.floor(relative_time / forecast_grid_size).astype(int)

    # Multiply the index by the bin_width to get the bin's start time
    bin_start_label = merged_meas_periods.period_start + (bin_index * forecast_grid_size)

    merged_meas_periods = (
        merged_meas_periods.groupby(["task_name", "name", bin_start_label.rename("bin_start")])[
            "value"
        ]
        .median()
        .reset_index()
    )
    merged_meas_periods["lower_bound"] = merged_meas_periods.name.map(lower_clinical_criteria)
    merged_meas_periods["upper_bound"] = merged_meas_periods.name.map(upper_clinical_criteria)

    merged_meas_periods["meets_criteria"] = (
        merged_meas_periods.value <= merged_meas_periods.upper_bound
    ) & (merged_meas_periods.value >= merged_meas_periods.lower_bound)

    period_labels = merged_meas_periods.groupby(["task_name"]).meets_criteria.all()
    return period_labels.astype(bool)


def get_last_period_basline_preds(daily_periods_df, period_labels, dominant_class=False):
    last_period_baseline_preds = (
        daily_periods_df.join(period_labels, on="task_name")
        # drops periods that weren't made into tasks
        .dropna()
        .sort_values(["pat_enc_csn_id", "period_start"])
        .assign(last_meets_criteria=lambda df: df.groupby("pat_enc_csn_id").meets_criteria.shift(1))
        .set_index("task_name")
        # fill first periods in encounter with dominant class, which is false
        .last_meets_criteria.fillna(0 if not dominant_class else 1)
        .astype(bool)
    )
    return last_period_baseline_preds


class DailyCriteriaTaskSet(TaskSet):
    def __init__(
        self,
        forecast_hours: int = 12,
        forecast_grid_size: int = 3,
        lookback_hours: int = 48,
        day_start_hour: int = 9,
        min_task_measurements=10,
        subset="test",
        route="simple-charts-dev",
        smoke_test=False,
        data_source="mimic4",
        refresh_cache=False,
        tight_criteria: bool = True,
        hourly_aggregate_labels: bool = False,
    ):
        self.forecast_hours = forecast_hours
        self.forecast_grid_size = forecast_grid_size
        self.lookback_hours = lookback_hours
        self.day_start_hour = day_start_hour
        self.tight_criteria = tight_criteria
        self.hourly_aggregate_labels = hourly_aggregate_labels

        data_source = create_data_source(data_source, route)

        self.adm_df = self.get_adm_df(data_source)
        self.abx_rx_df = data_source.get_abx_rx_df()

        self.daily_periods_df = get_daily_periods_from_adm(
            adm_df=self.adm_df,
            abx_rx_df=self.abx_rx_df,
            forecast_hours=forecast_hours,
            lookback_hours=lookback_hours,
            interval_hours=24.0,
            day_start_hour=day_start_hour,
        )

        # we want to make predictions at the middle of the bins
        forecast_grid = (
            B.range(np.float32, 0.0, forecast_hours, forecast_grid_size) + forecast_grid_size / 2
        )
        split_strategy = GriddedForecastSplitter(forecast_grid=forecast_grid)

        task_length_hours = lookback_hours + forecast_hours
        super().__init__(
            split_strategy,
            data_source,
            task_length_hours,
            min_task_measurements,
            subset,
            smoke_test,
            refresh_cache,
        )

        self.period_labels = get_period_criteria_labels(
            self.daily_periods_df,
            self.measurements_df,
            self.get_clinical_criteria(standardised=True),
            self.forecast_grid_size,
        )

    @staticmethod
    def get_adm_df(data_source):
        adm_df = data_source.get_adm_df()
        return adm_df.loc[adm_df.prescribed_iv]

    def get_joined_periods_labels(self):
        return self.daily_periods_df.join((self.period_labels), on="task_name")

    def _filter_measurements_to_iv(self):
        before_count = len(self.measurements_df.pat_enc_csn_id.unique())

        measurements_df = self.measurements_df.loc[
            self.measurements_df.pat_enc_csn_id.isin(
                self.adm_df.loc[self.adm_df.prescribed_iv].pat_enc_csn_id
            )
        ]
        after_count = len(measurements_df.pat_enc_csn_id.unique())

        logger.info(
            f"Filtered {before_count - after_count} of {before_count} non-IV abx encounters"
        )
        self.measurements_df = measurements_df

    def _get_tasks(self):
        self._filter_measurements_to_iv()
        return super()._get_tasks()

    def get_clinical_criteria(self, standardised=False):
        if self.tight_criteria:
            clinical_criteria = {
                "temp": (96.8, 100.4),
                "pulse": (41, 90),
                "resp_rate": (9, 20),
                "systolic_bp": (101, 219),
                "spo2": (94, np.inf),
            }

        else:
            clinical_criteria = {
                "temp": (96.8, 100.58),
                "pulse": (40, 131),
                "resp_rate": (8, 24),
                "systolic_bp": (90, 220),
                "spo2": (91, np.inf),
            }
        if standardised:
            for meas in clinical_criteria.keys():
                lower, upper = clinical_criteria[meas]
                std_params = self.std_params[meas]
                clinical_criteria[meas] = (
                    (lower - std_params["mean"]) / std_params["std"],
                    (upper - std_params["mean"]) / std_params["std"],
                )

        return clinical_criteria

    def generate_task_windows(self, enc_id, max_enc_time):
        """Generate regularly spaced task windows"""
        task_windows = []
        sub_period_df = self.daily_periods_df[
            ["pat_enc_csn_id", "task_name", "period_start", "period_end"]
        ].loc[self.daily_periods_df.pat_enc_csn_id == (enc_id)]

        task_index = 0
        for _, (enc_id, task_name, period_start, period_end) in sub_period_df.iterrows():
            task_windows.append(
                (task_name, (period_start - self.lookback_hours, period_start, period_end))
            )
            task_index += 1

        return task_windows

    def get_true_measurements_for_task(self, task_id, measurement_name):
        """Get the true values of the measurement in the prediction window, which is useful for plotting etc"""

        task_row = self.daily_periods_df.set_index("task_name").loc[task_id]
        period_start, period_end, enc_id = (
            task_row.period_start,
            task_row.period_end,
            task_row.pat_enc_csn_id,
        )

        return self.measurements_df.loc[
            (self.measurements_df.enc_elapsed_time > period_start)
            & (self.measurements_df.enc_elapsed_time < period_end)
            & (self.measurements_df.name == measurement_name)
            & (self.measurements_df.pat_enc_csn_id == enc_id)
        ].set_index("enc_elapsed_time")

    def get_cache_filename(self):
        """Generate cache filename for DailyCriteriaTaskSet."""
        data_source = getattr(self.data_source, "name", "unknown")
        route = getattr(self.data_source, "route", "unknown")
        # Note: max_hours_since_iv is hardcoded to 36 in get_daily_periods_from_adm
        # If this becomes configurable, it should be added to the cache filename
        return f"{data_source}_{route}_DailyCriteriaTaskSet_{self.subset}_{self.forecast_hours}f_{self.forecast_grid_size}g_{self.lookback_hours}h_{self.day_start_hour}start_{self.min_task_measurements}min.pkl"


def create_roc_pr_plot(y_true, y_pred_proba, title_prefix="Model"):
    """
    Create ROC and PR curves as subplots.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        title_prefix: Prefix for plot titles

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # ROC curve
    RocCurveDisplay.from_predictions(y_true, y_pred_proba, ax=ax1, plot_chance_level=True)
    ax1.set_title(f"{title_prefix} ROC Curve")

    # PR curve
    PrecisionRecallDisplay.from_predictions(y_true, y_pred_proba, ax=ax2, plot_chance_level=True)
    ax2.set_title("Precision-Recall Curve")

    plt.tight_layout()
    return fig


def create_calibration_plot(y_true, y_pred_proba, title_prefix="Model"):
    """
    Create calibration plot with histogram.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        title_prefix: Prefix for plot titles

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Calibration plot
    CalibrationDisplay.from_predictions(y_true, y_pred_proba, n_bins=10, strategy="uniform", ax=ax1)
    ax1.set_title("Calibration Plot")

    # Histogram of predicted probabilities
    ax2.hist(y_pred_proba, range=(0, 1), bins=30, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"{title_prefix} Prediction Distribution")

    plt.tight_layout()
    return fig


def check_values_meet_criteria(values, measurement_name, clinical_criteria, samples=False):
    """
    Check if values meet clinical criteria for a specific measurement.
    """
    lower, upper = clinical_criteria[measurement_name]
    return (values >= lower).all(axis=1 if samples else None) & (values <= upper).all(
        1 if samples else None
    )


def check_task_criteria(rescaled_predictions, measurement_names, clinical_criteria):
    """
    Check if all measurements in a task meet clinical criteria.
    """
    example_shape = rescaled_predictions[0].shape
    num_samples = example_shape[0] if len(example_shape) > 1 else 1
    meets_criteria_list = []
    for measurement_name, pred_values in zip(measurement_names, rescaled_predictions):
        # Skip empty measurements
        if pred_values.shape[-1] == 0:
            continue
        # Check if this measurement meets criteria
        meets_criteria = check_values_meet_criteria(
            pred_values, measurement_name, clinical_criteria, samples=num_samples > 1
        )
        meets_criteria_list.append(meets_criteria)

    if num_samples > 1:
        meets_criteria = B.mean(B.all(B.stack(*meets_criteria_list, axis=1), 1) * 1.0).item()
    else:
        meets_criteria = all(meets_criteria_list)

    return meets_criteria


def evaluate_all_task_predictions(task_preds, measurement_names, clinical_criteria):
    task_results = {}
    for task_id, pred_tensors in task_preds.items():
        task_meets_all_criteria = check_task_criteria(
            pred_tensors, measurement_names, clinical_criteria
        )
        task_results[task_id] = task_meets_all_criteria
    return task_results
