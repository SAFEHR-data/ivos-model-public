import warnings
from typing import Optional

import neuralprocesses as nps
import numpy as np
import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt

import lab as B
from primitivo_model.data.criteria import (
    DailyCriteriaTaskSet,
    evaluate_all_task_predictions,
)
from primitivo_model.nps.analysis import collect_mean_predictions
from primitivo_model.nps.data import TaskLoader

warnings.filterwarnings("ignore", category=UserWarning, module="neuralprocesses")


def collect_sample_predictions(model, batcher, num_samples, state):
    """Collect predictions for all tasks in the batcher."""
    model.eval()
    task_preds = {}

    with torch.no_grad():
        for batch in batcher.epoch():
            state, _, _, _, yt = nps.predict(
                state, model, batch["contexts"], batch["xt"], num_samples=num_samples
            )

            task_ids = batch["ids"]
            for task_id in task_ids:
                task_preds[task_id] = []
            for meas_idx, meas_pred in enumerate(yt):
                for i in range(meas_pred.shape[1]):
                    pred = meas_pred[:, i]
                    if batch["yt"] is not None:
                        pred = pred[:, ~B.isnan(batch["yt"][meas_idx][i])]

                    task_preds[task_ids[i]].append(pred)
    return task_preds, state


def compute_criteria_probability(model, batcher, state, criteria):
    model.eval()
    task_preds = {}

    with torch.no_grad():
        for batch in batcher.epoch():
            assert batch["yt"] is None
            xt = batch["xt"]
            state, mean, var, _, _ = nps.predict(state, model, batch["contexts"], xt)

            task_ids = batch["ids"]
            for idx in task_ids:
                task_preds[idx] = 0.0
            for meas_idx, (meas_mean_pred, meas_var_pred) in enumerate(zip(mean, var)):
                lower_bound, upper_bound = criteria[batcher.task_set.measurement_names[meas_idx]]
                for i in range(meas_mean_pred.shape[0]):
                    mean_pred = meas_mean_pred[i].cpu().numpy()
                    var_pred = meas_var_pred[i].cpu().numpy()

                    std_pred = np.sqrt(var_pred)

                    if len(mean_pred) == 0:
                        continue

                    # Compute probability that value is in [lower_bound, upper_bound]
                    # P(lower <= X <= upper) = P(X <= upper) - P(X < lower)
                    prob_upper = scipy.special.ndtr((upper_bound - mean_pred) / std_pred)
                    prob_lower = scipy.special.ndtr((lower_bound - mean_pred) / std_pred)
                    prob_in_range = prob_upper - prob_lower

                    log_prob_meets = np.log(prob_in_range)

                    # Accumulate log probabilities for this task
                    task_preds[task_ids[i]] += np.sum(log_prob_meets)

        for idx, log_prob in task_preds.items():
            task_preds[idx] = np.exp(log_prob)

    return task_preds, state


class CriteriaTaskLoader(TaskLoader):
    def __init__(
        self,
        task_set: DailyCriteriaTaskSet,
        batch_size: int,
        device,
        seed: int = 0,
        epoch_size: Optional[int] = None,
    ):
        if not isinstance(task_set, DailyCriteriaTaskSet):
            raise ValueError("CriteriaTaskLoader requires a DailyCriteriaTaskSet")

        super().__init__(task_set, batch_size, device, seed, epoch_size, no_yt=True)


def predict_meets_criteria_mean(state, model, batched_tasks: CriteriaTaskLoader):
    task_preds, state = collect_mean_predictions(model, batched_tasks, 1, state)
    criteria = batched_tasks.task_set.get_clinical_criteria(standardised=True)
    measurement_names = batched_tasks.task_set.measurement_names
    return pd.Series(evaluate_all_task_predictions(task_preds, measurement_names, criteria))


def predict_meets_criteria_samples(state, model, num_samples, batched_tasks: CriteriaTaskLoader):
    task_preds, state = collect_sample_predictions(model, batched_tasks, num_samples, state)
    criteria = batched_tasks.task_set.get_clinical_criteria(standardised=True)
    measurement_names = batched_tasks.task_set.measurement_names
    return pd.Series(evaluate_all_task_predictions(task_preds, measurement_names, criteria)), state


def predict_meets_criteria_exact(
    state, model, batched_tasks: CriteriaTaskLoader, grid_points_per_hour=None
):
    criteria = batched_tasks.task_set.get_clinical_criteria(standardised=True)
    task_preds, state = compute_criteria_probability(model, batched_tasks, state, criteria)
    return pd.Series(task_preds), state


def get_criteria_results_df(criteria_predictions: pd.Series, test_set):
    return pd.DataFrame(
        {"meets_criteria": test_set.period_labels, "prob_meets_criteria": criteria_predictions},
        index=test_set.period_labels.index,
    ).dropna()


def plot_clinical_criteria_predictions(
    task_ids: np.array, task_predictions: dict, daily_task_set: DailyCriteriaTaskSet
):
    fig, axes = plt.subplots(
        len(task_ids),
        len(daily_task_set.measurement_names),
        figsize=(len(daily_task_set.measurement_names) * 3, len(task_ids) * 2),
    )
    clinical_criteria = daily_task_set.get_clinical_criteria()

    for n, task_id in enumerate(task_ids):
        for i, measurement_name in enumerate(daily_task_set.measurement_names):
            pred_tensor = task_predictions[task_id][i]
            xt_tensor = daily_task_set.tasks[task_id]["xt"][i]
            is_samples = len(pred_tensor.shape) > 1

            # Skip if any tensor is empty
            true_measurements = daily_task_set.get_true_measurements_for_task(
                task_id, measurement_name
            )

            if true_measurements.empty:
                axes[n][i].text(
                    0.5,
                    0.5,
                    f"No data\nfor {measurement_name}",
                    ha="center",
                    va="center",
                    transform=axes[n][i].transAxes,
                )
                axes[n][i].set_title(f"{measurement_name}")
                continue

            if is_samples:
                pred_tensor = pred_tensor.T
            pred_values = pred_tensor.squeeze(1).cpu().numpy()
            time_values = xt_tensor.squeeze()

            # Denormalize using std_params
            mean_val = daily_task_set.std_params[measurement_name]["mean"]
            std_val = daily_task_set.std_params[measurement_name]["std"]
            pred_values = pred_values * std_val + mean_val
            true_values = true_measurements.value * std_val + mean_val

            # Create time series plot
            axes[n][i].plot(
                true_measurements.index, true_values, "o-", label="True", alpha=0.7, markersize=4
            )
            axes[n][i].plot(
                time_values,
                pred_values,
                "s-",
                c="green",
                alpha=0.03 if is_samples else 0.7,
                markersize=2,
            )

            axes[n][i].axhline(
                y=clinical_criteria[measurement_name][0],
                color="red",
                linestyle="--",
                alpha=0.6,
                label="Lower limit",
            )
            axes[n][i].axhline(
                y=clinical_criteria[measurement_name][1],
                color="red",
                linestyle="--",
                alpha=0.6,
                label="Upper limit",
            )

            axes[n][i].set_xlabel("Time (hours)")
            if i == 0:
                axes[n][i].set_ylabel(f" {task_id}")
            # Add criteria results to title
            title = f"{measurement_name}"

            axes[n][i].set_title(title)
            if n == 0 and i == 0:
                axes[n][i].legend()
            axes[n][i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
