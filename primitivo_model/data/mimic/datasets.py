import numpy as np
import pandas as pd
from loguru import logger

from primitivo_model.db import BaseDb

resonable_values = {
    "pulse": (10, 400),
    # "diastolic_bp": (0, s250),
    "systolic_bp": (0, 400),
    "spo2": (60, 100),
    "resp_rate": (1, 99),
    "temp": (50.0, 120.0),
    # "neutrophils": (0, 450),
    # "crp": (0, 600),
    # "whitecell": (0, 500),
}

charts_variables = {
    "Temperature Fahrenheit": "temp",
    "Non Invasive Blood Pressure systolic": "systolic_bp",
    # "Non Invasive Blood Pressure diastolic": "diastolic_bp",
    "O2 saturation pulseoxymetry": "spo2",
    "Respiratory Rate": "resp_rate",
    "Heart Rate": "pulse",
}

labs_variables = {
    # "Neutrophils": "neutrophils",
    "White Blood Cells": "whitecell",
    # "C-Reactive Protein": "crp",
}


def create_simple_charts_dataset(db: BaseDb, dataset_db: BaseDb, add_labs: bool = False) -> None:
    """
    Create a dataset from the processed MIMIC database.

    Args:
        db: The processed MIMIC database
    """
    logger.info("Creating dataset from processed MIMIC database...")

    charts_df = db.read(
        f"SELECT * FROM charts as c WHERE c.label IN {tuple(charts_variables.keys())}"
    )
    logger.info(
        f"Loaded {len(charts_df)} chart events from {charts_df['subject_id'].nunique()} patients"
    )
    charts_df = charts_df.drop(columns=["valueuom"]).assign(
        label=charts_df.label.replace(charts_variables)
    )
    if add_labs:
        labs_df = db.read(
            f"SELECT * FROM labs as l WHERE l.label IN {tuple(labs_variables.keys())}"
        )
        logger.info(
            f"Loaded {len(labs_df)} lab events from {labs_df['subject_id'].nunique()} patients"
        )

        labs_df = labs_df.assign(label=labs_df.label.replace(labs_variables))
        lab_chart_df = pd.concat([labs_df, charts_df])
    else:
        lab_chart_df = charts_df

    # Convert charttime to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(lab_chart_df["charttime"]):
        lab_chart_df["charttime"] = pd.to_datetime(lab_chart_df["charttime"])

    # logger.info(f"loaded charts with variables {charts_df.label.unique()}")

    # Get admission time for each patient-admission
    adm_df = db.read("select * from admissions")
    logger.info(f"Number of admissions: {len(adm_df)}")

    # first_times = lab_chart_df.groupby(["subject_id", "hadm_id"])["charttime"].min().reset_index()
    # first_times.rename(columns={"charttime": "first_charttime"}, inplace=True)

    # Merge first_times back to the main dataframe
    lab_chart_df = pd.merge(lab_chart_df, adm_df[["hadm_id", "admittime"]], on=["hadm_id"])

    # Calculate hours since admission
    lab_chart_df["hours_since_admission"] = (
        lab_chart_df["charttime"] - lab_chart_df["admittime"]
    ).dt.total_seconds() / 3600
    lab_chart_df = lab_chart_df.drop(columns=["admittime"])

    min_hours = 24
    max_hours = lab_chart_df.groupby(["subject_id", "hadm_id"])["hours_since_admission"].max()
    eligible_admissions = max_hours[max_hours >= min_hours].reset_index()[["subject_id", "hadm_id"]]

    logger.info(
        f"Found {len(eligible_admissions)} patient admissions with at least {min_hours} hours of data"
    )

    max_enc_length = 14 * 24
    charts_filtered = pd.merge(
        lab_chart_df, eligible_admissions, on=["subject_id", "hadm_id"], how="inner"
    )
    charts_filtered = charts_filtered[charts_filtered["hours_since_admission"] <= max_enc_length]

    logger.info(
        f"After filtering: {len(charts_filtered)} chart events from {charts_filtered['subject_id'].nunique()} patients"
    )

    charts_filtered = charts_filtered.rename(
        columns={
            "hours_since_admission": "time",
            # "min_real_admittime": "admttime",
            "valuenum": "value",
            "label": "label",
            "hadm_id": "admission_id",
        }
    )
    charts_filtered = charts_filtered[["admission_id", "time", "value", "label"]]

    logger.info(f"{charts_filtered.duplicated().sum()} duplicate chart rows found")

    na_values = charts_filtered.value.isna()
    logger.info(f"Dropping {na_values.sum()} NA values...")
    charts_filtered = charts_filtered.loc[~na_values]
    # remove outliers
    clean_groups = []
    # Group by label and identify outliers
    for label, group in charts_filtered.groupby("label"):
        mean = group["value"].mean()
        std = group["value"].std()
        # threshold = 4 * std

        total_rows_before = len(group)

        # Count outliers for this label
        outlier_mask = (group["value"] > resonable_values[label][1]) | (
            group["value"] < resonable_values[label][0]
        )
        outlier_count = outlier_mask.sum()

        clean_group = group[~outlier_mask]
        clean_groups.append(clean_group)

        if outlier_count > 0:
            clean_mean = clean_group["value"].mean()
            clean_std = clean_group["value"].std()

            logger.info(
                f"Removed {outlier_count}/{total_rows_before} rows from {label} (µ={mean:.1f} -> {clean_mean:.1f}, s={std:.1f} -> {clean_std:.1f})"
            )
        else:
            logger.info(
                f"No outliers from {label} (µ={mean:.1f}, s={std:.1f}, max={clean_group['value'].max():.1f})"
            )

    charts_filtered = pd.concat(clean_groups)
    charts_filtered = charts_filtered.sort_values("time")

    logger.info(f"Saving measurements for {len(charts_filtered.admission_id.unique())} admissions")

    # Save filtered charts
    dataset_db.save("measurements", charts_filtered)


def get_iv_abx_adm(db: BaseDb, dataset_db: BaseDb):
    abx_df = db.read("SELECT * FROM antibiotics").drop(columns=["stay_id"])

    # keep only relevant and standardise names
    relevant_routes = ["IV", "PO/NG", "PO", "ORAL"]
    abx_df = abx_df.loc[abx_df.route.isin(relevant_routes)].assign(
        route=abx_df.route.replace({"PO/NG": "PO", "ORAL": "PO"})
    )

    stays_with_iv = abx_df.loc[abx_df["route"] == "IV", "hadm_id"].unique()
    stays_with_oral = abx_df.loc[abx_df["route"] == "PO", "hadm_id"].unique()
    stays_with_both = set(stays_with_iv).intersection(set(stays_with_oral))

    adm_df = db.read("select * from admissions")

    total_stays = adm_df["hadm_id"].nunique()
    num_iv = len(stays_with_iv)
    num_oral = len(stays_with_oral)
    num_both = len(stays_with_both)
    logger.info(
        f"Stays with IV: {num_iv} ({num_iv / total_stays:.1%}), "
        f"Oral: {num_oral} ({num_oral / total_stays:.1%}), "
        f"Both: {num_both} ({num_both / total_stays:.1%})"
    )

    adm_df = adm_df.assign(prescribed_iv=adm_df["hadm_id"].isin(stays_with_iv))
    anchor_group_start = adm_df.anchor_year_group.str.split().apply(lambda x: x[0]).astype(int)
    time_since_anchor = adm_df.admittime - pd.to_datetime(adm_df.anchor_year.astype(str) + "-01-01")

    min_years_since_anchor = adm_df.admittime.dt.year - adm_df.anchor_year
    real_age = min_years_since_anchor + adm_df.anchor_age

    adm_df = adm_df.assign(
        min_real_admittime=pd.to_datetime(anchor_group_start.astype(str) + "-01-01")
        + time_since_anchor,
        age=real_age,
    )
    adm_df = adm_df[
        [
            "hadm_id",
            "subject_id",
            "admittime",
            "dischtime",
            "deathtime",
            "min_real_admittime",
            "prescribed_iv",
            "race",
            "age",
            "gender",
        ]
    ]

    logger.info(
        f"{adm_df['prescribed_iv'].mean():.1%} of {len(adm_df)} encoutners prescribed IV antibiotics"
    )

    charts_df = dataset_db.read("select * from measurements")
    encounters_with_measurements = charts_df.admission_id.unique()
    to_drop_idx = adm_df.hadm_id.isin(encounters_with_measurements)
    logger.info(f"Dropped {(~to_drop_idx).sum()} encounters with no measurements.")
    # drop encounters with no measurements
    adm_df = adm_df.loc[to_drop_idx]

    diag_df = db.read("select * from diagnoses")

    # Create diagnosis flags for each encounter (hadm_id)
    # Check if each encounter had any diagnoses for sepsis, pneumonia, or UTI
    diag_df["has_sepsis"] = diag_df["long_title"].str.lower().str.contains("sepsis", na=False)
    diag_df["has_uti"] = (
        diag_df["long_title"]
        .str.lower()
        .str.contains("urinary tract infection|pyelonephritis", na=False)
    )
    diag_df["has_pneumonia"] = diag_df["long_title"].str.lower().str.contains("pneumonia", na=False)

    # Group by hadm_id and get max value (any True becomes True for the encounter)
    diag_summary = (
        diag_df.groupby("hadm_id")[["has_sepsis", "has_uti", "has_pneumonia"]].max().reset_index()
    )

    adm_df = pd.merge(
        adm_df,
        diag_summary[["hadm_id", "has_sepsis", "has_uti", "has_pneumonia"]],
        on="hadm_id",
    )
    print(adm_df)

    dataset_db.save("iv_abx_adm", adm_df)


def get_abx_rx(db: BaseDb, dataset_db: BaseDb):
    abx_df = db.read("SELECT * FROM antibiotics").drop(columns=["stay_id"])
    # keep only relevant and standardise names
    relevant_routes = ["IV", "PO/NG", "PO", "ORAL"]
    abx_df = abx_df.loc[abx_df.route.isin(relevant_routes)].assign(
        route=abx_df.route.replace({"PO/NG": "PO", "ORAL": "PO"})
    )

    adm_df = db.read("select * from admissions")
    abx_df = pd.merge(abx_df, adm_df[["hadm_id", "admittime"]], on=["hadm_id"])
    abx_df["starttime"] = (abx_df["starttime"] - abx_df["admittime"]).dt.total_seconds() / 3600
    abx_df["stoptime"] = (abx_df["stoptime"] - abx_df["admittime"]).dt.total_seconds() / 3600

    abx_df["antibiotic"] = abx_df["antibiotic"].str.lower()
    logger.info(f"Num antibiotic names after lowercasing: {len(abx_df['antibiotic'].unique())}")
    logger.info(f"Number of antibiotic prescriptions: {len(abx_df)}")
    num_iv = (abx_df["route"] == "IV").sum()
    num_po = (abx_df["route"] == "PO").sum()
    logger.info(f"Number of IV prescriptions: {num_iv}")
    logger.info(f"Number of PO prescriptions: {num_po}")

    adm_df = dataset_db.read("select * from iv_abx_adm")
    encounters_with_measurements = adm_df.hadm_id.unique()
    to_drop_idx = abx_df.hadm_id.isin(encounters_with_measurements)
    logger.info(f"Dropped {(~to_drop_idx).sum()} prescriptions with no info.")
    # drop encounters with no measurements
    abx_df = abx_df.loc[to_drop_idx]

    dataset_db.save("abx_rx", abx_df)


def create_bolton_ivos_labels(db: BaseDb, dataset_db: BaseDb):
    logger.info("Creating IVOS from processed MIMIC database...")

    abx_df = db.read("SELECT * FROM antibiotics")

    # this drops all the non-ICU patients as in Bolton
    abx_df = abx_df[~abx_df.stay_id.isna()]

    # keep only relevant and standardise names
    relevant_routes = ["IV", "PO/NG", "PO", "NU", "ORAL"]
    abx_df = abx_df.loc[abx_df.route.isin(relevant_routes)].assign(
        route=abx_df.route.replace({"PO/NG": "PO", "NU": "PO", "ORAL": "PO"})
    )
    # if patient has IV, PO, IV
    # this just gives the span of all IV prescriptions in whole stay
    antibiotic_courses = (
        abx_df.groupby(["stay_id", "route"])
        .agg(starttime=("starttime", "min"), stoptime=("stoptime", "max"))
        .reset_index()
    )

    # Find stays that have both IV and PO routes
    stays_with_iv = antibiotic_courses.loc[antibiotic_courses["route"] == "IV", "stay_id"].unique()
    stays_with_po = antibiotic_courses.loc[antibiotic_courses["route"] == "PO", "stay_id"].unique()
    iv_and_po_stays = set(stays_with_iv).intersection(stays_with_po)

    # Filter for those stays
    filtered_abx = antibiotic_courses[antibiotic_courses["stay_id"].isin(iv_and_po_stays)]

    # Pivot the table to a wide format and create specific columns for IV and PO times
    ivos_df = filtered_abx.pivot(index="stay_id", columns="route", values=["starttime", "stoptime"])
    # Flatten the multi-level column index and format names
    ivos_df.columns = [f"{route.lower()}_{value}" for value, route in ivos_df.columns]

    # Convert datetime columns to just the date
    for col in ["iv_starttime", "iv_stoptime", "po_starttime", "po_stoptime"]:
        ivos_df[col] = pd.to_datetime(ivos_df[col]).dt.date

    ivos_df = ivos_df.reset_index()

    logger.info(f"Found {ivos_df.stay_id.nunique()} stays with both IV and PO antibiotics.")
    ivos_df = ivos_df.loc[ivos_df["iv_stoptime"] <= ivos_df["po_stoptime"]]
    logger.info(f"Found {ivos_df.stay_id.nunique()} stays after filtering for IV->PO switch.")

    ivos_df = ivos_df.assign(
        iv_duration=(ivos_df["iv_stoptime"] - ivos_df["iv_starttime"]) / np.timedelta64(1, "D"),
        po_duration=(ivos_df["po_stoptime"] - ivos_df["po_starttime"]) / np.timedelta64(1, "D"),
        antibiotic_gap=(ivos_df["po_starttime"] - ivos_df["iv_stoptime"]) / np.timedelta64(1, "D"),
    )
    ivos_df = ivos_df.query("(iv_duration >= 0) & (iv_duration <= 7) & (po_duration >= 0)")
    logger.info(
        f"Found {ivos_df.stay_id.nunique()} stays after filtering for long or short durations"
    )

    ivos_df = ivos_df.assign(
        total_duration=ivos_df.po_duration + ivos_df.iv_duration + ivos_df.antibiotic_gap
    )

    iv_dates = ivos_df.assign(
        date=ivos_df.apply(
            lambda r: pd.date_range(start=r["iv_starttime"], end=r["iv_stoptime"], freq="D"), axis=1
        )
    ).explode("date")[["stay_id", "date"]]
    iv_dates["iv_flag"] = 1

    # Create a similar DataFrame for PO courses
    po_dates = ivos_df.assign(
        date=ivos_df.apply(
            lambda r: pd.date_range(start=r["po_starttime"], end=r["po_stoptime"], freq="D"), axis=1
        )
    ).explode("date")[["stay_id", "date"]]
    po_dates["first_po_flag"] = 1

    # Merge the two DataFrames to get flags for each day
    daily_flags = pd.merge(iv_dates, po_dates, on=["stay_id", "date"], how="outer").sort_values(
        by=["stay_id", "date"]
    )

    daily_flags = daily_flags.sort_values(by=["stay_id", "date"])[
        ["stay_id", "date", "iv_flag", "first_po_flag"]
    ]

    # end of MIMIC notebook 1.1
    # implements po_label_fun
    daily_flags["po_flag"] = (daily_flags["iv_flag"] != 1).astype(int)

    # this implements implements iv_treatment_length_fun
    # note that in the notebook, the function accounds (incorrectly?) for multiple
    # periods of IV thearpy, but this is not possible given the earlier setup
    daily_flags["iv_treatment_length"] = daily_flags.groupby("stay_id").cumcount()

    # Set non-IV days to nan
    daily_flags.loc[daily_flags["iv_flag"] != 1, "iv_treatment_length"] = np.nan
    daily_flags = daily_flags.drop(columns=["iv_flag", "first_po_flag"])

    # Create a mask to identify the first non-po_flag row for each stay_id
    first_non_po = daily_flags.groupby("stay_id")["po_flag"].transform(
        lambda x: (x != 1).cumsum() > 0
    )

    # Filter the data to keep only rows from the first non-po_flag row onwards
    daily_flags = daily_flags[first_non_po].reset_index(drop=True)

    logger.info(f"Processed daily flags for {daily_flags.stay_id.nunique()} stays.")

    # print(abx_df.groupby("route").subject_id.count())
    dataset_db.save("bolton_switch_labels", daily_flags)
