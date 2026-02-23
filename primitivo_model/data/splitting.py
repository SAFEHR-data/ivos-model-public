from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

from primitivo_model.config import settings


def split_tasks_ids(tasks):
    """
    Split task IDs into train, evaluation and test sets.

    Args:
        tasks: Dictionary of tasks or list of task IDs

    Returns:
        train_ids, val_ids, test_ids: Lists containing the split task IDs
    """
    assert abs(settings.TRAIN_RATIO + settings.VAL_RATIO + settings.TEST_RATIO - 1.0) < 1e-6, (
        "Ratios must sum to 1"
    )

    # Set random seed for reproducibility, do not change!
    np.random.seed(42)

    # Get list of task IDs and shuffle
    if isinstance(tasks, dict):
        task_ids = list(tasks.keys())
    else:
        task_ids = list(tasks)
    np.random.shuffle(task_ids)

    # Calculate split indices
    n_tasks = len(task_ids)
    n_train = int(n_tasks * settings.TRAIN_RATIO)
    n_val = int(n_tasks * settings.VAL_RATIO)

    # Split task IDs
    train_ids = task_ids[:n_train]
    val_ids = task_ids[n_train : n_train + n_val]
    test_ids = task_ids[n_train + n_val :]

    logger.info(
        f"Split {n_tasks} task IDs into {len(train_ids)} train, {len(val_ids)} val, and {len(test_ids)} test"
    )

    return train_ids, val_ids, test_ids


def compute_standardization_params(df):
    """
    Compute standardization parameters (mean, std) for each measurement type.

    Args:
        df: DataFrame containing measurements

    Returns:
        Dictionary of standardization parameters
    """
    standardization_params = {}

    # Compute parameters for each measurement type separately
    for name in df.name.unique():
        # Select rows for this measurement type
        mask = df.name == name
        values = df.loc[mask, "value"]

        # Calculate mean and standard deviation
        mean_val = values.mean()
        std_val = values.std()

        # Handle case where std is 0 to avoid division by zero
        if std_val == 0:
            std_val = 1.0
            logger.warning(f"Standard deviation is 0 for {name}, using std=1.0")

        # Store parameters
        standardization_params[name] = {"mean": mean_val, "std": std_val}

    return standardization_params


def standardize_with_params(df, params):
    """
    Standardize a DataFrame using pre-computed standardization parameters.

    Args:
        df: DataFrame to standardize
        params: Dictionary with standardization parameters {name: {'mean': mean, 'std': std}}

    Returns:
        Standardized DataFrame
    """
    result_df = df.copy()

    for name, param in params.items():
        mask = result_df.name == name
        if not any(mask):
            continue

        mean_val = param["mean"]
        std_val = param["std"]

        result_df.loc[mask, "value"] = (result_df.loc[mask, "value"] - mean_val) / std_val

    return result_df


def split_tasks_by_time(df: pd.DataFrame, cutoff_date, remaining_val_ratio: float):
    """Split encounters by admission time cutoff into train/val/test."""
    logger.info(f"Time‐based split enabled, using cutoff date: {cutoff_date}")

    if "admittime" not in df.columns:
        raise ValueError("DataFrame missing 'admittime' column for time‐based splitting")
    cutoff = pd.Timestamp(cutoff_date)
    first_adm = df.groupby("pat_enc_csn_id")["admittime"].min()
    test_ids = first_adm[first_adm >= cutoff].index.tolist()
    remaining = first_adm[first_adm < cutoff].index.tolist()

    np.random.seed(42)
    np.random.shuffle(remaining)
    n_train = int(len(remaining) * (1 - remaining_val_ratio))
    train_ids = remaining[:n_train]
    val_ids = remaining[n_train:]
    return train_ids, val_ids, test_ids


def get_split_and_standardise(
    measurements_df: pd.DataFrame, subset: str, cutoff_date=None
) -> Tuple[pd.DataFrame, dict]:
    task_ids = measurements_df.pat_enc_csn_id.unique()

    if cutoff_date is not None:
        train_ids, val_ids, test_ids = split_tasks_by_time(measurements_df, cutoff_date, 0.1)
    else:
        train_ids, val_ids, test_ids = split_tasks_ids(task_ids)

    # Split measurements dataframe by encounter IDs
    train_df = measurements_df[measurements_df.pat_enc_csn_id.isin(train_ids)]
    val_df = measurements_df[measurements_df.pat_enc_csn_id.isin(val_ids)]
    test_df = measurements_df[measurements_df.pat_enc_csn_id.isin(test_ids)]

    logger.info(
        f"Split measurements: {len(train_ids)} ({len(train_df)}) train, {len(val_ids)} ({len(val_df)}) val, {len(test_ids)} ({len(test_df)}) test "
    )
    # Compute standardization parameters from training data only
    std_params = compute_standardization_params(train_df)
    # Log the standardization parameters for train subset
    if subset == "train":
        for name, params in std_params.items():
            logger.info(f"Standardizing {name}: mean={params['mean']:.4f}, std={params['std']:.4f}")

    # Apply standardization to all datasets using the same parameters
    train_df_std = standardize_with_params(train_df, std_params)
    val_df_std = standardize_with_params(val_df, std_params)
    test_df_std = standardize_with_params(test_df, std_params)

    # Split into training, cross-validation, and evaluation data.
    if subset == "train":
        measurements_subset = train_df_std
    elif subset == "val":
        measurements_subset = val_df_std
    elif subset == "test":
        measurements_subset = test_df_std
    else:
        raise ValueError(f'Unknown subset "{subset}"')

    return measurements_subset, std_params
