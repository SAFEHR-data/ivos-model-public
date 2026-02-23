import pandas as pd

from primitivo_model.data.criteria import (
    get_daily_periods_from_adm,
    get_last_period_basline_preds,
    get_period_criteria_labels,
    get_period_start,
)


def test_get_period_start_after_start_hour():
    """Test when the timestamp is after the day_start_hour."""
    timestamp = pd.Timestamp("2023-01-01 10:00:00")
    day_start_hour = 9
    expected = pd.Timestamp("2023-01-01 09:00:00")
    assert get_period_start(timestamp, day_start_hour) == expected


def test_get_period_start_before_start_hour():
    """Test when the timestamp is before the day_start_hour."""
    timestamp = pd.Timestamp("2023-01-01 08:00:00")
    day_start_hour = 9
    expected = pd.Timestamp("2022-12-31 09:00:00")
    assert get_period_start(timestamp, day_start_hour) == expected


def test_get_period_start_at_start_hour():
    """Test when the timestamp is exactly the day_start_hour."""
    timestamp = pd.Timestamp("2023-01-01 09:00:00")
    day_start_hour = 9
    expected = pd.Timestamp("2023-01-01 09:00:00")
    assert get_period_start(timestamp, day_start_hour) == expected


def test_get_period_start_at_midnight():
    """Test when the timestamp is at midnight."""
    timestamp = pd.Timestamp("2023-01-01 00:00:00")
    day_start_hour = 9
    expected = pd.Timestamp("2022-12-31 09:00:00")
    assert get_period_start(timestamp, day_start_hour) == expected


def test_get_daily_periods_from_adm():
    """Test the get_daily_periods_from_adm function."""
    data = {
        "pat_enc_csn_id": [1],
        "admittime": [pd.Timestamp("2023-01-01 10:00:00")],
        "dischtime": [pd.Timestamp("2023-01-04 12:00:00")],
    }
    adm_df = pd.DataFrame(data)

    # Create fake antibiotic prescription data
    abx_data = {
        "pat_enc_csn_id": [1, 1],
        "route": ["IV", "PO"],
        "starttime": [24.0, 48.0],  # Hours since admission
        "stoptime": [48.0, 72.0],   # Hours since admission
    }
    abx_rx_df = pd.DataFrame(abx_data)

    periods_df = get_daily_periods_from_adm(
        adm_df,
        abx_rx_df,
        forecast_hours=12.0,
        lookback_hours=48.0,
        interval_hours=24.0,
        day_start_hour=9,
    )

    assert len(periods_df) == 1
    assert "task_name" in periods_df.columns
    assert periods_df["period_start"].iloc[0] == 71.0
    assert periods_df["period_end"].iloc[0] == 83.0
    assert periods_df["task_name"].iloc[0] == "1n3"


def test_get_period_criteria_labels():
    """Test the get_period_criteria_labels function."""
    daily_periods_data = {
        "pat_enc_csn_id": [1, 1],
        "task_name": ["task1", "task2"],
        "period_start": [0, 24],
        "period_end": [12, 36],
    }
    daily_periods_df = pd.DataFrame(daily_periods_data)

    measurements_data = {
        "pat_enc_csn_id": [1, 1, 1, 1],
        "name": ["temp", "pulse", "temp", "pulse"],
        "enc_elapsed_time": [5, 6, 28, 30],
        "value": [98.0, 80.0, 101.0, 90.0],
    }
    measurements_df = pd.DataFrame(measurements_data)

    clinical_criteria = {"temp": (96.8, 100.4), "pulse": (0, 100)}

    period_labels = get_period_criteria_labels(daily_periods_df, measurements_df, clinical_criteria, 3)

    assert isinstance(period_labels, pd.Series)
    assert len(period_labels) == 2
    assert period_labels.loc["task1"]
    assert not period_labels.loc["task2"]


def test_get_last_period_basline_preds():
    """Test the get_last_period_basline_preds function."""
    daily_periods_data = {
        "pat_enc_csn_id": [1, 1, 2, 2],
        "task_name": ["task1", "task2", "task3", "task4"],
        "period_start": [0, 24, 0, 24],
    }
    daily_periods_df = pd.DataFrame(daily_periods_data)

    period_labels_data = {
        "task_name": ["task1", "task2", "task3", "task4"],
        "meets_criteria": [True, False, True, True],
    }
    period_labels = pd.Series(
        period_labels_data["meets_criteria"],
        index=period_labels_data["task_name"],
        name="meets_criteria",
    )

    baseline_preds = get_last_period_basline_preds(daily_periods_df, period_labels)

    assert isinstance(baseline_preds, pd.Series)
    assert len(baseline_preds) == 4
    assert not baseline_preds.loc["task1"]  # No previous period, defaults to False
    assert baseline_preds.loc["task2"]
    assert not baseline_preds.loc["task3"]  # No previous period for this patient
    assert baseline_preds.loc["task4"]
