import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import neuralprocesses.torch as nps
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

import lab as B
from primitivo_model.data.util import (
    check_context_empty,
    get_xt_extremal_values,
)
from primitivo_model.mlflow_utils import log_plot_to_mlflow
from primitivo_model.naive.model import predict as predict_naive
from primitivo_model.tabular.evaluation import (
    collect_task_regression_predictions,
    extract_measurement_category_from_one_hot,
)
from primitivo_model.util import to_numpy

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


class IVSwitchingAnalysisPlotter:
    """Class for creating IV switching analysis plots for patient encounters."""

    def __init__(self):
        tab10 = plt.get_cmap("tab10")

        self.color_blue = tab10(0)  # Blue
        self.color_orange = tab10(1)  # Orange
        self.color_green = tab10(2)  # Green
        self.color_red = tab10(3)  # Red
        self.color_purple = tab10(4)  # Purple
        self.color_brown = tab10(5)  # Brown
        self.color_pink = tab10(6)  # Pink
        self.color_gray = tab10(7)  # Gray
        self.color_olive = tab10(8)  # Olive
        self.color_cyan = tab10(9)  # Cyan

        # Color map for different measurements
        self.measurement_color_map = {
            "temp": self.color_red,
            "pulse": self.color_blue,
            "resp_rate": self.color_green,
            "systolic_bp": self.color_purple,
            "spo2": self.color_cyan,
        }

        # Color map for routes
        self.route_color_map = {"IV": self.color_blue, "PO": self.color_green}

        # Colors for labels panel
        self.label_colors = {
            "meets_criteria": self.color_red,
            "on_iv": self.color_blue,
            "on_po": self.color_green,
        }

        self.prediction_color = self.color_red
        self.threshold_color = self.color_red

        # Display names for measurements
        self.measurement_display_names = {
            "temp": "Temp.",
            "pulse": "HR",
            "resp_rate": "RR",
            "systolic_bp": "SBP",
            "spo2": r"$\text{SpO}_2$",
        }

        # Display names for labels
        self.label_display_names = {
            "meets_criteria": "Meets criteria?",
            "on_iv": "On IV?",
            "on_po": "On PO?",
        }

        # Antibiotic name shortenings (case-insensitive matching)
        self.antibiotic_shortenings = {
            "sulfameth/trimethoprim suspension": "Sulfameth/Trimeth",
            "sulfameth/trimethoprim ds": "Sulfameth/Trimeth",
            "piperacillin-tazobactam": "Pip.-Taz.",
            "erythromycin ethylsuccinate suspension": "Erythromycin",
        }

    def plot_iv_switching_analysis(
        self,
        pat_enc_csn_id,
        joined_periods_preds_labels,
        test_set,
        figsize=(15, 12),
    ):
        """
        Create a comprehensive plot showing IV switching analysis for a patient encounter.

        Args:
            pat_enc_csn_id: Patient encounter ID
            joined_periods_preds_labels: DataFrame with predictions and labels
            test_set: DailyCriteriaTaskSet object containing measurements and criteria
            figsize: Figure size tuple

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        patient_periods = joined_periods_preds_labels[
            joined_periods_preds_labels["pat_enc_csn_id"] == pat_enc_csn_id
        ].sort_values("period_start")

        patient_measurements = test_set.measurements_df[
            test_set.measurements_df["pat_enc_csn_id"] == pat_enc_csn_id
        ]

        patient_abx = test_set.abx_rx_df[test_set.abx_rx_df["pat_enc_csn_id"] == pat_enc_csn_id]

        if patient_periods.empty:
            print(f"No data found for patient {pat_enc_csn_id}")
            return None

        measurement_names = test_set.measurement_names

        # Calculate number of panels: 1 (predictions) + 1 (labels) + len(vitals) + 1 (antibiotics)
        n_panels = 2 + len(measurement_names) + 1

        # Create height ratios: prediction panel (0.5), labels panel (0.5), vital panels (0.5 each), antibiotic panel (1)
        height_ratios = [0.5, 0.5] + [0.5] * len(measurement_names) + [1]

        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, height_ratios=height_ratios)

        if n_panels == 1:
            axes = [axes]

        # Collect legend handles and labels from all panels
        all_handles = []
        all_labels = []

        # Panel 1: Model predictions
        prob_handles, prob_labels = self._plot_probability_panel(axes[0], patient_periods)
        all_handles.extend(prob_handles)
        all_labels.extend(prob_labels)

        # Panel 2: Categorical labels
        label_handles, label_labels = self._plot_labels_panel(axes[1], patient_periods)
        all_handles.extend(label_handles)
        all_labels.extend(label_labels)

        # Panels 3 to n-1: Individual vital signs
        for i, meas_name in enumerate(measurement_names):
            # Only collect threshold legend from the first vital sign panel
            threshold_handles, threshold_labels = self._plot_individual_vital_panel(
                axes[i + 2], patient_measurements, test_set, meas_name, return_legend=i == 0
            )
            if i == 0:  # Only add threshold legend once
                all_handles.extend(threshold_handles)
                all_labels.extend(threshold_labels)

        # Last panel: Antibiotic timeline
        abx_handles, abx_labels = self._plot_antibiotics_panel(
            axes[-1], patient_abx, pat_enc_csn_id
        )
        all_handles.extend(abx_handles)
        all_labels.extend(abx_labels)

        # Add period start/end lines to all panels
        self._add_period_lines(axes, patient_periods)

        # Align x-axes and set common time range
        if not patient_measurements.empty:
            time_min = min(
                patient_measurements["enc_elapsed_time"].min(),
                patient_periods["period_start"].min() if not patient_periods.empty else 0,
            )
            time_max = max(
                patient_measurements["enc_elapsed_time"].max(),
                patient_periods["period_end"].max() if not patient_periods.empty else 0,
            )

            # Add some padding
            time_range = time_max - time_min
            time_min -= time_range * 0.05
            time_max += time_range * 0.05

            for ax in axes:
                ax.set_xlim(time_min, time_max)

        # Set common x-label only on bottom plot
        axes[-1].set_xlabel("Time (hours since admission)")
        for ax in axes[:-1]:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        # Create a single combined legend at the top of the figure
        unique_labels = {}
        for handle, label in zip(all_handles, all_labels):
            if label and label not in unique_labels:
                unique_labels[label] = handle

        if unique_labels:
            fig.legend(
                unique_labels.values(),
                unique_labels.keys(),
                loc="upper center",
                bbox_to_anchor=(0.6, 0.97),
                ncol=min(len(unique_labels), 7),  # Up to 5 columns
                fontsize=9,
                frameon=True,
            )

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Adjusted to make room for legend

        return fig

    def _plot_probability_panel(self, ax, patient_periods):
        """Plot predicted probabilities"""
        period_midpoints = (patient_periods["period_start"] + patient_periods["period_end"]) / 2

        line = ax.plot(
            period_midpoints,
            patient_periods["prob_meets_criteria"],
            "-o",
            color=self.prediction_color,
            label="Predicted Probability",
            linewidth=2,
            markersize=6,
        )

        # Add probability values as text annotations
        for i, (midpoint, prob) in enumerate(
            zip(period_midpoints, patient_periods["prob_meets_criteria"])
        ):
            ax.annotate(
                f"{prob:.3f}",
                xy=(midpoint, prob),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=self.prediction_color,
                ha="left",
                va="bottom",
            )

        ax.set_ylabel("Predicted\nProbability")
        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(axis="y")

        return line, ["Predicted Probability"]

    def _plot_labels_panel(self, ax, patient_periods):
        """Plot categorical labels"""
        label_categories = ["meets_criteria", "on_iv", "on_po"]
        label_markers = {
            "meets_criteria": "s",
            "on_iv": "^",
            "on_po": "D",
        }  # square, triangle, diamond

        handles = []
        labels = []

        for i, label_name in enumerate(label_categories):
            if label_name in patient_periods.columns:
                positive_periods = patient_periods[patient_periods[label_name]]
                if not positive_periods.empty:
                    positive_midpoints = (
                        positive_periods["period_start"] + positive_periods["period_end"]
                    ) / 2
                    display_name = self.label_display_names.get(label_name, label_name)
                    scatter = ax.scatter(
                        positive_midpoints,
                        [i] * len(positive_midpoints),
                        color=self.label_colors[label_name],
                        s=50,
                        marker=label_markers[label_name],
                        label=display_name,
                        zorder=5,
                        alpha=0.8,
                    )
                    handles.append(scatter)
                    labels.append(display_name)

        ax.set_yticks(range(len(label_categories)))
        ax.set_yticklabels([self.label_display_names.get(lc, lc) for lc in label_categories])
        ax.set_ylim(-0.5, len(label_categories) - 0.5)

        return handles, labels

    def _add_period_lines(self, axes, patient_periods):
        """Add faint vertical lines to show period starts and ends"""
        period_boundaries = set()
        for _, row in patient_periods.iterrows():
            period_boundaries.add(row["period_start"])
            period_boundaries.add(row["period_end"])

        for ax in axes:
            for boundary in period_boundaries:
                ax.axvline(
                    x=boundary, color="grey", linestyle=":", alpha=0.3, linewidth=0.8, zorder=0
                )

    def _plot_individual_vital_panel(
        self, ax, patient_measurements, test_set, meas_name, return_legend=False
    ):
        """Plot a single vital sign with clinical criteria thresholds"""
        # Get clinical criteria (non-standardized)
        clinical_criteria = test_set.get_clinical_criteria(standardised=False)

        line_color = self.measurement_color_map.get(meas_name, "blue")

        meas_data = patient_measurements[patient_measurements["name"] == meas_name]

        display_name = self.measurement_display_names.get(meas_name, meas_name)

        if meas_data.empty:
            ax.text(
                0.5,
                0.5,
                f"No {display_name} data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(display_name)
            return [], []

        # Denormalize the values
        try:
            if hasattr(test_set, "std_params") and meas_name in test_set.std_params:
                mean_val = test_set.std_params[meas_name]["mean"]
                std_val = test_set.std_params[meas_name]["std"]
                denorm_values = meas_data["value"] * std_val + mean_val
            else:
                denorm_values = meas_data["value"]
        except (KeyError, AttributeError):
            denorm_values = meas_data["value"]

        ax.plot(
            meas_data["enc_elapsed_time"],
            denorm_values,
            "o-",
            color=line_color,
            alpha=0.7,
            markersize=3,
            linewidth=1,
        )

        handles = []
        labels = []

        # Add criteria thresholds; only add to legend if return_legend is True
        if meas_name in clinical_criteria:
            lower, upper = clinical_criteria[meas_name]
            if lower != -np.inf:
                line = ax.axhline(
                    y=lower,
                    color=self.threshold_color,
                    linestyle="--",
                    alpha=0.8,
                    linewidth=1,
                )
                if return_legend:
                    handles.append(line)
                    labels.append("Clinical threshold")
            if upper != np.inf:
                line = ax.axhline(
                    y=upper,
                    color=self.threshold_color,
                    linestyle="--",
                    alpha=0.8,
                    linewidth=1,
                )
                # Only add to legend if we didn't already add a lower threshold
                if return_legend and lower == -np.inf:
                    handles.append(line)
                    labels.append("Clinical threshold")

        ax.set_ylabel(display_name)
        ax.tick_params(axis="y", labelsize=8)

        return handles, labels

    def _plot_antibiotics_panel(self, ax, patient_abx, pat_enc_csn_id):
        """Plot antibiotic timeline"""
        if patient_abx.empty:
            ax.text(
                0.5,
                0.5,
                "No antibiotic data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Antibiotic Timeline")
            return [], []

        def format_antibiotic_name(abx_name):
            abx_lower = abx_name.lower()
            for key, short_name in self.antibiotic_shortenings.items():
                if key in abx_lower:
                    return short_name
            return abx_name.capitalize()

        # Get unique antibiotics and create y-positions with shortened names
        unique_abx = patient_abx["antibiotic"].unique()
        y_labels = sorted([format_antibiotic_name(abx) for abx in unique_abx])
        y_pos = {label: i for i, label in enumerate(y_labels)}

        # Create reverse mapping from original name to display name
        abx_to_display = {abx: format_antibiotic_name(abx) for abx in unique_abx}

        # Track which route labels have been added to avoid duplicates
        route_labels_added = set()

        for _, row in patient_abx.iterrows():
            start = row["starttime"]
            end = (
                row["stoptime"] if pd.notna(row["stoptime"]) else start + 24
            )  # Assume 24h if no end

            route = row["route"]
            label = route if route not in route_labels_added else ""
            if route not in route_labels_added:
                route_labels_added.add(route)

            # Use hatching patterns to distinguish IV (no hatch) from PO (hatched)
            # This helps with color-blind accessibility
            hatch_pattern = None if route == "IV" else "///"  # PO gets diagonal hatching

            ax.barh(
                y=y_pos[abx_to_display[row["antibiotic"]]],
                width=end - start,
                left=start,
                height=0.6,
                color=self.route_color_map.get(route, "grey"),
                edgecolor="black",
                alpha=0.8,
                label=label,
                hatch=hatch_pattern,
            )

        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(list(y_pos.keys()))
        ax.invert_yaxis()
        ax.set_ylabel("Antibiotic")

        # Get legend handles and labels without duplicates
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            by_label = dict(zip(labels, handles))
            return list(by_label.values()), list(by_label.keys())

        return [], []


def setup_plot_task(task):
    """
    Extracts contexts, xt, yt, min_x, and max_x from a task dictionary.
    """
    contexts = task["contexts"]
    xt = task["xt"]
    yt = task["yt"]
    max_x = max(get_xt_extremal_values(xt, max_or_min="max"))
    min_x = min(get_xt_extremal_values(xt, max_or_min="min"))
    return contexts, xt, yt, min_x, max_x


def to_numpy_list(data):
    return [to_numpy(d, squeeze=True) for d in data]


def plot_task_np(model, task, measurement_names, num_samples=5, hours_back=4):
    contexts, xt, yt, min_x, max_x = setup_plot_task(task)

    if check_context_empty(contexts):
        return go.Figure()

    with torch.no_grad():
        x_test = B.to_active_device(torch.linspace(min_x, max_x, 24).reshape(1, 1, -1))

        contexts_cast = []
        for x_ctx, y_ctx in contexts:
            x_ctx_cast = B.cast(torch.float32, x_ctx)
            y_ctx_cast = B.cast(torch.float32, y_ctx)
            contexts_cast.append((x_ctx_cast, y_ctx_cast))

        y_mean, y_var, _, _ = nps.predict(
            model,
            contexts_cast,
            nps.AggregateInput(*((x_test, i) for i in range(len(measurement_names)))),
            batch_size=1,
        )

        _, _, ft_samples, yt_samples = nps.ar_predict(
            model,
            contexts_cast,
            nps.AggregateInput(*((x_test, i) for i in range(len(measurement_names)))),
            num_samples=16,
            order="given",
        )

    y_mean = to_numpy_list(y_mean)
    y_var = to_numpy_list(y_var)
    yt = to_numpy_list(yt)
    xt = to_numpy_list(xt)
    ft_samples = to_numpy_list(ft_samples)
    x_test = to_numpy_list(x_test)
    contexts = [(to_numpy(xc), to_numpy(yc)) for xc, yc in contexts]

    return create_measurement_plot(
        contexts,
        xt,
        yt,
        y_mean,
        y_var,
        ft_samples,
        x_test,
        measurement_names,
        hours_back=hours_back,
    )


def plot_task_tabular(model, feature_extractor, task, measurement_names, hours_back=4):
    contexts, xt, yt, min_x, max_x = setup_plot_task(task)

    if check_context_empty(contexts):
        return go.Figure()

    x_test = [
        np.linspace(min_x, max_x, 100).reshape(1, 1, -1) for _ in range(len(measurement_names))
    ]
    feature_df = feature_extractor.get_task_features(task["contexts"], x_test, min_x)
    y_mean_flat = model.predict(feature_df)

    # Extract measurement categories from one-hot encoding
    measurement_categories = extract_measurement_category_from_one_hot(
        feature_df, measurement_names
    )

    y_mean = collect_task_regression_predictions(
        y_mean_flat, measurement_categories, len(measurement_names)
    )

    y_mean = [y.squeeze() for y in y_mean]
    return create_measurement_plot(
        contexts,
        xt,
        yt,
        y_mean,
        None,
        None,
        x_test,
        measurement_names,
        hours_back=hours_back,
    )


def plot_task_naive(task, measurement_names, hours_back=4):
    contexts, xt, yt, min_x, max_x = setup_plot_task(task)

    if check_context_empty(contexts):
        return go.Figure()

    x_test = [
        B.to_active_device(torch.linspace(min_x, max_x, 100).reshape(1, 1, -1))
        for _ in range(len(measurement_names))
    ]

    y_mean, _, *_ = predict_naive(
        contexts,
        x_test,
    )

    y_mean = [y.squeeze() for y in y_mean]
    return create_measurement_plot(
        contexts,
        xt,
        yt,
        y_mean,
        None,
        None,
        x_test,
        measurement_names,
        hours_back=hours_back,
    )


def create_measurement_plot(
    contexts,
    xt,
    yt,
    y_mean,
    y_var,
    y_samples,
    x_test,
    measurement_names,
    plot_title=None,
    hours_back=4,
):
    """
    Create a matplotlib figure showing measurements, predictions, and uncertainty.

    Args:
        contexts: List of (x_ctx, y_ctx) tuples per measurement
        xt: List of x_target tensors per measurement
        yt: List of y_target tensors per measurement
        y_mean: Prediction means from the model
        y_var: Prediction variances from the model
        y_samples: Samples from the model's posterior
        x_test: The x values used for prediction
        measurement_names: List of measurement names
        plot_title: Optional title for the plot
        hours_back: Number of hours before earliest target point to show on x-axis, None for default

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_measurements = len(measurement_names)
    n_cols = int(np.ceil(np.sqrt(n_measurements)))
    n_rows = int(np.ceil(n_measurements / n_cols))

    fig = plt.figure(figsize=(3 * n_cols, 2 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    if plot_title:
        fig.suptitle(plot_title)

    x_pred = x_test[0].squeeze()

    # Calculate x-axis limits if hours_back is provided
    x_limits = None
    if hours_back is not None:
        earliest_x = float("inf")
        latest_x = float("-inf")

        for x_trg in xt:
            valid_x = x_trg[~np.isnan(x_trg)]
            if len(valid_x) > 0:
                earliest_x = min(earliest_x, np.min(valid_x))
                latest_x = max(latest_x, np.max(valid_x))

        if earliest_x != float("inf"):
            x_limits = [earliest_x - hours_back, latest_x + 0.5]

    legend_elements = []
    legend_labels = []

    for idx, measurement_name in enumerate(measurement_names):
        row = idx // n_cols
        col = idx % n_cols

        ax = fig.add_subplot(gs[row, col])
        ax.set_title(measurement_name)

        x_ctx, y_ctx = contexts[idx]
        x_ctx_np = x_ctx.squeeze()
        y_ctx_np = y_ctx.squeeze()

        x_trg = xt[idx]
        y_trg = yt[idx]

        # Filter out NaN values in target (from padding)
        valid_indices = ~np.isnan(y_trg)
        x_trg = x_trg[valid_indices]
        y_trg = y_trg[valid_indices]

        # Add context points (shown as blue circles)
        context_plot = ax.scatter(
            x_ctx_np,
            y_ctx_np,
            color="blue",
            s=64,
            label="Context Points" if idx == 0 else "_nolegend_",
        )

        if idx == 0:
            legend_elements.append(context_plot)
            legend_labels.append("Context Points")

        # Add target points (shown as red triangles)
        target_plot = ax.scatter(
            x_trg,
            y_trg,
            color="red",
            s=100,
            marker="^",
            label="Target Points" if idx == 0 else "_nolegend_",
        )

        if idx == 0:
            legend_elements.append(target_plot)
            legend_labels.append("Target Points")

        y_pred_mean = y_mean[idx]
        pred_line = ax.plot(
            x_pred,
            y_pred_mean,
            color="green",
            linewidth=2,
            label="Model Prediction" if idx == 0 else "_nolegend_",
        )[0]

        if idx == 0:
            legend_elements.append(pred_line)
            legend_labels.append("Model Prediction")

        if y_var:
            y_pred_std = B.sqrt(y_var[idx])

            # Add uncertainty bands (2 standard deviations)
            uncertainty = ax.fill_between(
                x_pred,
                y_pred_mean - 2 * y_pred_std,
                y_pred_mean + 2 * y_pred_std,
                color="green",
                alpha=0.2,
                label="95% Confidence" if idx == 0 else "_nolegend_",
            )

            if idx == 0:
                legend_elements.append(uncertainty)
                legend_labels.append("95% Confidence")

        # Add model sample predictions (semi-transparent lines)
        if y_samples:
            num_samples = y_samples[idx].shape[0]
            for j in range(num_samples):
                y_sample = to_numpy(y_samples[idx][j])
                sample_line = ax.plot(
                    x_pred,
                    y_sample,
                    color="black",
                    alpha=0.15,
                    linewidth=1,
                    label="Sample" if (idx == 0 and j == 0) else "_nolegend_",
                )[0]

                if idx == 0 and j == 0:
                    legend_elements.append(sample_line)
                    legend_labels.append("Sample")

        is_bottom_row = row == n_rows - 1 or idx >= n_measurements - n_cols
        if is_bottom_row:
            ax.set_xlabel("Time (hours)")

        # Only add y labels to the leftmost column
        if col == 0:
            ax.set_ylabel("Standardized Value")

        if x_limits:
            ax.set_xlim(x_limits)

    fig.legend(
        legend_elements,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(legend_labels),
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the legend

    return fig


def make_and_log_task_plots(plot_task_fn, num_tasks_to_plot, task_set, run_id, fig_name=None):
    num_plot = min(num_tasks_to_plot, len(task_set))
    for i, task_id in enumerate(task_set.encounter_ids):
        fig = plot_task_fn(
            task_set[task_id],
            task_set.measurement_names,
        )
        if not fig_name:
            fig_name = "predictions_task"
        fig_path = f"{fig_name}_{task_id}.png"
        log_plot_to_mlflow(fig, fig_path)

        if i >= num_plot:
            break


color_map = {"IV": "blue", "PO": "green"}


def plot_abx_timeline(df, hadm_id, switch_dates=None, legend=True, big_plot=True):
    """Plots the antibiotic timeline for a given hadm_id.

    Args:
        df: DataFrame containing antibiotic data
        hadm_id: Hospital admission ID to plot
        switch_dates: Optional dict of dates to highlight on the plot
                     {label: datetime object or string}
    """
    encounter_df = df[df["hadm_id"] == hadm_id].copy()
    return plot_abx_timeline_from_encounter_df(
        encounter_df, hadm_id, switch_dates, legend, big_plot
    )


def plot_abx_timeline_stay(df, stay_id, switch_dates=None, legend=True, big_plot=True):
    encounter_df = df[df["stay_id"] == stay_id].copy()
    return plot_abx_timeline_from_encounter_df(
        encounter_df, stay_id, switch_dates, legend, big_plot
    )


def plot_abx_timeline_from_encounter_df(
    encounter_df, id, switch_dates=None, legend=True, big_plot=True
):
    y_labels = sorted(encounter_df["antibiotic"].unique())
    y_pos = {label: i for i, label in enumerate(y_labels)}

    if not y_pos:
        return  # Don't plot if no data

    fig, ax = plt.subplots(
        figsize=(6, (len(y_labels) * (0.4 if big_plot else 0.2)) + (1.0 if big_plot else 0.8))
    )

    for _, row in encounter_df.iterrows():
        start = row["starttime"]
        # If stoptime is NaT, assume it's a single day treatment for plotting
        end = row["stoptime"] if pd.notna(row["stoptime"]) else start + pd.Timedelta(days=1)

        ax.barh(
            y=y_pos[row["antibiotic"]],
            width=end - start,
            left=start,
            height=0.6,
            color=color_map.get(row["route"], "grey"),
            edgecolor="black",
            label=row["route"],
        )

    # Highlight switch dates if provided
    if switch_dates is not None:
        # Handle both dict and single date for backward compatibility
        if not isinstance(switch_dates, dict):
            switch_dates = {"Switch Date": switch_dates}

        switch_colors = ["red", "orange", "purple", "brown", "pink"]

        for i, (label, switch_date) in enumerate(switch_dates.items()):
            if not switch_date:
                continue

            color = switch_colors[i % len(switch_colors)]
            ax.axvline(
                x=switch_date, color=color, linestyle=":", linewidth=2, label=label, alpha=0.8
            )

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))
    ax.invert_yaxis()

    if not isinstance(start, float):
        ax.set_xlabel("Date")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
    else:
        ax.set_xlabel("Time (days)")

    ax.set_title(f"stay_id = {id}")

    # Create a legend without duplicates
    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    return fig
