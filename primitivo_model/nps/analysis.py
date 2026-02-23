import neuralprocesses as nps
import numpy as np
import torch

import lab.torch as B
from primitivo_model.mlflow_utils import log_dataframe_to_mlflow
from primitivo_model.util import to_numpy


def collect_mean_predictions(model, batcher, state):
    """Collect predictions for all tasks in the batcher."""
    model.eval()
    task_preds = {}

    with torch.no_grad():
        for batch in batcher.epoch():
            state, mean_pred, _, _, _ = nps.predict(state, model, batch["contexts"], batch["xt"])

            task_ids = batch["ids"]
            for id in task_ids:
                task_preds[id] = []

            for meas_idx, meas_pred in enumerate(mean_pred):
                for i in range(len(meas_pred)):
                    pred = meas_pred[i]
                    pred = pred[~B.isnan(batch["yt"][meas_idx][i])]
                    task_preds[task_ids[i]].append(pred)

    return task_preds, state


def prepare_measurement_data(task_preds, test_tasks, use_standardized=True):
    """Prepare per-measurement error and horizon arrays from task predictions."""
    measurement_data = {}
    for meas_idx in range(test_tasks.num_measurements):
        measurement_data[meas_idx] = {"errors": [], "horizons": []}

    for task_id in task_preds.keys():
        task = test_tasks.tasks[task_id]
        pred = task_preds[task_id]

        for meas_idx, meas_name in enumerate(test_tasks.measurement_names):
            x_ctx = task["contexts"][meas_idx][0]
            yt = task["yt"][meas_idx]

            if B.shape(x_ctx)[-1] == 0 or B.shape(yt)[-1] == 0:
                continue

            max_x_ctx = B.max(x_ctx)

            pred_val = to_numpy(pred[meas_idx])
            target_val = yt

            if not use_standardized and hasattr(test_tasks, "std_params"):
                if meas_name in test_tasks.std_params:
                    mean_val = test_tasks.std_params[meas_name]["mean"]
                    std_val = test_tasks.std_params[meas_name]["std"]
                    pred_val = pred_val * std_val + mean_val
                    target_val = target_val * std_val + mean_val

            example_error = B.abs(target_val - pred_val)
            relative_horizon = task["xt"][meas_idx] - max_x_ctx

            measurement_data[meas_idx]["errors"].append(example_error[0, 0])
            measurement_data[meas_idx]["horizons"].append(relative_horizon[0, 0])

    for meas_idx in range(test_tasks.num_measurements):
        if measurement_data[meas_idx]["errors"]:
            measurement_data[meas_idx]["errors"] = B.concat(*measurement_data[meas_idx]["errors"])
            measurement_data[meas_idx]["horizons"] = B.concat(
                *measurement_data[meas_idx]["horizons"]
            )

    return measurement_data


def create_mean_error_summary(measurement_data, test_tasks, n_bootstrap=1000, random_seed=42):
    """Create a bootstrapped MAE summary table per measurement type."""
    import pandas as pd

    summary_data = []
    rng = np.random.RandomState(random_seed)

    for meas_idx in range(test_tasks.num_measurements):
        errors = to_numpy(measurement_data[meas_idx]["errors"])
        count = len(errors)

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(errors, size=count, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)

        summary_data.append(
            {
                "Measurement": test_tasks.measurement_names[meas_idx],
                "median": np.median(bootstrap_means),
                "ci_lower": np.percentile(bootstrap_means, 5),
                "ci_upper": np.percentile(bootstrap_means, 95),
                "count": count,
            }
        )

    return pd.DataFrame(summary_data).set_index("Measurement")


def log_raw_mae_summary(task_preds, test_tasks, run_id):
    """Compute and log the raw measurement MAE summary artifact."""
    raw_maes = prepare_measurement_data(task_preds, test_tasks, use_standardized=False)
    df_raw_meas_mae = create_mean_error_summary(raw_maes, test_tasks)
    log_dataframe_to_mlflow(df_raw_meas_mae, "raw_measurement_mae_summary.csv", run_id)
