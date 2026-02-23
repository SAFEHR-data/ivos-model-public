"""
MIMIC dataset processing module.

This module provides utilities to process the MIMIC-IV database.
"""

import numpy as np
import pandas as pd
from loguru import logger

from primitivo_model.db import BaseDb


def preprocess_mimic_data(db: BaseDb, output_db: BaseDb, smoke_test=False):
    """
    Process the MIMIC database and create a processed version.

    Args:
        db: Source MIMIC database
        output_db: Processed MIMIC database (output destination)
        smoke_test: If True, limit to a small subset of admissions
    """
    admissions_df = get_admissions(db)
    if smoke_test:
        admissions_df = admissions_df.iloc[:3000]  # only use 3000 admissions for smoke test

    logger.info(f"Loaded admissions with {len(admissions_df)} rows")
    output_db.save("admissions", admissions_df)

    abx_df = get_antibiotics(db, output_db)
    output_db.save("antibiotics", abx_df)

    charts_with_labels = get_charts(db, output_db)
    output_db.save("charts", charts_with_labels)

    # Process and save lab events
    labs_df = get_lab_events(db, output_db)
    output_db.save("labs", labs_df)

    diag_df = get_diagnoses(db, output_db)
    output_db.save("diagnoses", diag_df)


def get_admissions(db: BaseDb) -> pd.DataFrame:
    query = """
        SELECT 
            a.*,
            p.anchor_age,
            p.anchor_year,
            p.anchor_year_group,
            p.gender
        FROM 
            mimiciv_hosp.admissions a
        JOIN 
            mimiciv_hosp.patients p 
            ON a.subject_id = p.subject_id
        WHERE 
            a.hadm_id IN (SELECT DISTINCT hadm_id FROM mimiciv_icu.icustays)
            AND p.anchor_age IS NOT NULL
        ORDER BY a.subject_id, a.hadm_id;
        """
    cohort_df = db.read(query)
    return cohort_df


def get_diagnoses(db: BaseDb, output_db: BaseDb) -> pd.DataFrame:
    adm_ids_df = output_db.read("SELECT hadm_id FROM admissions")
    logger.info(f"Filtering diagnoses for {len(adm_ids_df)} admissions")

    query = """
    SELECT 
        d.*, 
        c.long_title,
    FROM 
        mimiciv_hosp.diagnoses_icd d 
    JOIN 
        mimiciv_hosp.d_icd_diagnoses c 
        ON d.icd_code = c.icd_code
    WHERE 
        d.hadm_id IN (SELECT hadm_id FROM adm_ids_df)
    """
    diag_df = db.read_with_dataframes(query, {"adm_ids_df": adm_ids_df})
    return diag_df


def get_charts(db: BaseDb, output_db: BaseDb) -> pd.DataFrame:
    """
    Process chart events from MIMIC database, filtering by admissions.

    Args:
        db: Source MIMIC database
        output_db: Processed MIMIC database

    Returns:
        Processed chart events DataFrame
    """
    # Get admission IDs from the processed database
    adm_ids_df = output_db.read("SELECT hadm_id FROM admissions")
    logger.info(f"Filtering chart events for {len(adm_ids_df)} admissions")

    # Create a single query that filters chartevents and joins with d_items
    full_query = """
    SELECT 
        ce.subject_id, 
        ce.hadm_id, 
        ce.charttime, 
        ce.itemid, 
        ce.valuenum, 
        ce.valueuom,
        di.label
    FROM 
        mimiciv_icu.chartevents ce
    JOIN
        mimiciv_icu.d_items di
        ON ce.itemid = di.itemid
    WHERE 
        ce.hadm_id IN (SELECT hadm_id FROM adm_ids_df)
    """

    logger.info("Processing chart events - this may take some time...")
    charts_with_labels = db.read_with_dataframes(full_query, {"adm_ids_df": adm_ids_df})

    logger.info(f"Loaded {len(charts_with_labels)} chart events")
    logger.info(f"Number of unique patients: {charts_with_labels['subject_id'].nunique()}")

    # Get count of unique subjects per label
    pat_for_item = (
        charts_with_labels.groupby("label")["subject_id"].nunique().sort_values(ascending=False)
    )

    # Get counts of each label
    label_counts = charts_with_labels["label"].value_counts()

    # Get labels that are both in top 100 by patient and top 200 by count
    frequent_labels1 = pat_for_item[:100]
    frequent_labels2 = label_counts.head(200)

    # Find intersection of both sets
    frequent_labels = frequent_labels1.loc[frequent_labels1.index.isin(frequent_labels2.index)]
    logger.info(f"Selected {len(frequent_labels)} frequent labels")

    # Filter to keep only frequent labels
    mask_frequent_labels = charts_with_labels["label"].isin(frequent_labels.index)
    charts_filtered = charts_with_labels.loc[mask_frequent_labels].copy()

    logger.info(f"After filtering by frequent labels: {len(charts_filtered)} rows")

    # Drop rows with any NaN values
    charts_cleaned = charts_filtered.dropna()
    unq_labels = charts_cleaned.label.unique()

    logger.info(f"Kept {len(unq_labels)} labels: \n {unq_labels}")
    logger.info(f"Saving {len(charts_with_labels)} chart events")
    logger.info(f"Number of unique patients: {charts_with_labels['subject_id'].nunique()}")
    return charts_cleaned


def verify_rate_calculation(
    df, rate_unit, seconds_per_duration, amount_multiplier=1.0, weight_factor=False
):
    # Check that rate * duration calculations match the amount values using a helper function

    temp_df = df.loc[(df["rate"].notnull()) & (df["rateuom"].str.contains(rate_unit))].copy()
    if len(temp_df) == 0:
        logger.info(f"No rows to verify for rate {rate_unit}")
        return

    duration_hours = (
        temp_df["endtime"] - temp_df["starttime"]
    ).dt.total_seconds() / seconds_per_duration

    temp_df["computed_amount"] = temp_df["rate"] * duration_hours
    if weight_factor:
        temp_df["computed_amount"] *= temp_df["patientweight"]

    # Compare with recorded amount (with appropriate unit conversion)
    mismatches = len(
        temp_df.loc[abs(temp_df["computed_amount"] / amount_multiplier - temp_df["amount"]) > 0.01]
    )
    if mismatches > 0:
        logger.warning(
            f"Found {mismatches} records where {rate_unit} rate calculations don't match recorded amounts"
        )


def get_input_events(db: BaseDb, output_db: BaseDb) -> pd.DataFrame:
    """
    Process input events from MIMIC database, filtering by admissions.

    Args:
        db: Source MIMIC database
        output_db: Processed MIMIC database

    Returns:
        Processed input events DataFrame
    """
    # Get admission IDs from the processed database
    adm_ids_df = output_db.read("SELECT hadm_id FROM admissions")
    logger.info(f"Filtering input events for {len(adm_ids_df)} admissions")

    # Create a single query that filters inputevents by admission IDs
    # and selects only columns of interest
    full_query = """
    SELECT 
        ie.subject_id,
        ie.hadm_id,
        ie.starttime,
        ie.endtime,
        ie.itemid,
        ie.amount,
        ie.amountuom,
        ie.rate,
        ie.rateuom,
        ie.patientweight,
        ie.ordercategorydescription,
        di.label
    FROM 
        mimiciv_icu.inputevents ie
    JOIN
        mimiciv_icu.d_items di
        ON ie.itemid = di.itemid
    WHERE 
        ie.hadm_id IN (SELECT hadm_id FROM adm_ids_df)
    """

    logger.info("Processing input events - this may take some time...")
    inputs_with_labels = db.read_with_dataframes(full_query, {"adm_ids_df": adm_ids_df})

    logger.info(f"Loaded {len(inputs_with_labels)} input events")
    logger.info(f"Number of unique patients: {inputs_with_labels['subject_id'].nunique()}")

    # List of medications/fluids to retain
    retained_list = [
        "Albumin 5%",
        "Dextrose 5%",
        "Lorazepam (Ativan)",
        "Calcium Gluconate",
        "Midazolam (Versed)",
        "Phenylephrine",
        "Furosemide (Lasix)",
        "Norepinephrine",
        "Magnesium Sulfate",
        "Nitroglycerin",
        "Insulin - Glargine",
        "Insulin - Humalog",
        "Insulin - Regular",
        "Heparin Sodium",
        "Morphine Sulfate",
        "Potassium Chloride",
        "Packed Red Blood Cells",
        "Gastric Meds",
        "D5 1/2NS",
        "LR",
        "K Phos",
        "Solution",
        "Sterile Water",
        "Metoprolol",
        "Piggyback",
        "OR Crystalloid Intake",
        "OR Cell Saver Intake",
        "PO Intake",
        "GT Flush",
        "KCL (Bolus)",
        "Magnesium Sulfate (Bolus)",
    ]

    # Filter to keep only medications/fluids in the retained list
    mask_retained_labels = inputs_with_labels["label"].isin(retained_list)
    inputs_filtered = inputs_with_labels.loc[mask_retained_labels].copy()

    logger.info("Cleaning medications/fluids to ensure consistent units...")

    # These inputs have some bad units in the data, we need to remove those values
    unit_requirements = {
        "Heparin Sodium (Prophylaxis)": "dose",
        "Magnesium Sulfate": "grams",
        "Metoprolol": "mg",
        "D5 1/2NS": "mL",
        "LR": "mL",
        "OR Crystalloid Intake": "mL",
        "PO Intake": "mL",
        "Gastric Meds": "mL",
        "GT Flush": "mL",
        "Phenylephrine": "mg",
        "Potassium Chloride": "mEq",
    }

    # Apply all unit filters at once
    for medication, required_unit in unit_requirements.items():
        before_count = len(inputs_filtered)
        # Create mask for rows to keep (either not the target medication, or has the correct unit)
        mask = ~(
            (inputs_filtered["label"] == medication)
            & (inputs_filtered["amountuom"] != required_unit)
        )
        inputs_filtered = inputs_filtered.loc[mask]
        removed = before_count - len(inputs_filtered)
        if removed > 0:
            logger.info(
                f"Removed {removed} {medication} records with incorrect unit (keeping only {required_unit})"
            )

    # Define the expected rate units for each medication based on frequency
    rate_unit_requirements = {
        "Dextrose 5%": "mL/hour",
        "Furosemide (Lasix)": "mg/hour",
        "Magnesium Sulfate (Bolus)": "mL/hour",
        "Norepinephrine": "mcg/kg/min",
        "Packed Red Blood Cells": "mL/hour",
        "Phenylephrine": "mcg/kg/min",
        "Piggyback": "mL/hour",
        "Solution": "mL/hour",
        "Sterile Water": "mL/hour",
        "Heparin Sodium": "units/hour",
    }
    for medication, required_unit in rate_unit_requirements.items():
        before_count = len(inputs_filtered)
        # Create mask for rows to keep:
        # - Either not the target medication
        # - Or it's the medication but rate is NaN (keep these)
        # - Or it's the medication with the correct rate unit
        mask = (
            (inputs_filtered["label"] != medication)
            | (inputs_filtered["rate"].isna())
            | (inputs_filtered["rateuom"] == required_unit)
        )
        inputs_filtered = inputs_filtered.loc[mask]
        removed = before_count - len(inputs_filtered)
        if removed > 0:
            logger.info(
                f"Removed {removed} {medication} records with incorrect rate (keeping only {required_unit})"
            )

    # Verify all rate unit types
    verify_rate_calculation(
        inputs_filtered, "mcg/kg/hour", 3600, amount_multiplier=1000, weight_factor=True
    )
    verify_rate_calculation(inputs_filtered, "mL/hour", 3600)
    verify_rate_calculation(inputs_filtered, "mL/hour", 3600)
    verify_rate_calculation(inputs_filtered, "mg/hour", 3600)
    verify_rate_calculation(inputs_filtered, "mcg/hour", 3600)
    verify_rate_calculation(inputs_filtered, "units/hour", 3600)
    verify_rate_calculation(inputs_filtered, "mg/min", 60)
    verify_rate_calculation(
        inputs_filtered, "mcg/kg/min", 60, amount_multiplier=1000, weight_factor=True
    )

    # Define duration threshold for splitting records
    duration_split_hours = 0.5

    # Split the dataset based only on duration
    duration_mask = (inputs_filtered["endtime"] - inputs_filtered["starttime"]) > pd.Timedelta(
        hours=duration_split_hours
    )

    # 1. Long duration records (to be split)
    df_long = inputs_filtered.loc[duration_mask].copy().reset_index(drop=True)
    # 2. Short duration records (keep as is)
    df_short = inputs_filtered.loc[~duration_mask].copy().reset_index(drop=True)

    # Verify the split was complete
    assert len(df_long) + len(df_short) == len(inputs_filtered)

    # Process long-duration records by splitting into multiple entries
    def split_records(df):
        if len(df) == 0:
            return df

        # Calculate number of repeats needed based on duration
        df["repeat_count"] = np.ceil(
            (df["endtime"] - df["starttime"]).dt.total_seconds() / (3600 * duration_split_hours)
        ).astype(int)

        # Create expanded dataframe with repeated rows
        df_expanded = df.loc[df.index.repeat(df["repeat_count"])].copy()

        # Generate evenly spaced timestamps for each original record
        df_expanded["charttime"] = df_expanded.groupby(level=0)["starttime"].transform(
            lambda x: pd.date_range(
                start=x.iat[0], freq=f"{int(60 * duration_split_hours)}min", periods=len(x)
            )
        )

        # Divide the amount by the number of repeats
        df_expanded["amount"] = df_expanded["amount"] / df_expanded["repeat_count"]

        return df_expanded

    logger.info("Splitting long duration doses, this can take a while...")
    # Process long duration records
    df_long_expanded = split_records(df_long)

    # For short-duration records, just copy the starttime to charttime
    df_short["charttime"] = df_short["starttime"]

    # Combine all processed dataframes
    inputs_processed = pd.concat([df_long_expanded, df_short], sort=True)

    logger.info(f"Split long-duration records: {len(inputs_filtered)} → {len(inputs_processed)}")
    logger.info(f"Saving {len(inputs_processed)} chart events")
    logger.info(f"Number of unique patients: {inputs_processed['subject_id'].nunique()}")
    return inputs_processed


def get_lab_events(db: BaseDb, output_db: BaseDb) -> pd.DataFrame:
    adm_ids_df = output_db.read("SELECT hadm_id FROM admissions")
    logger.info(f"Filtering lab events for {len(adm_ids_df)} admissions")

    full_query = """
    SELECT 
        le.subject_id, 
        le.hadm_id, 
        le.charttime, 
        le.valuenum, 
        le.itemid,
        dl.label
    FROM 
        mimiciv_hosp.labevents le
    JOIN
        mimiciv_hosp.d_labitems dl
        ON le.itemid = dl.itemid
    WHERE 
        le.hadm_id IN (SELECT hadm_id FROM adm_ids_df)
    """

    logger.info("Processing lab events - this may take some time...")
    labs_df = db.read_with_dataframes(full_query, {"adm_ids_df": adm_ids_df})

    logger.info(f"Loaded {len(labs_df)} lab events")

    # Get only top 150 most used tests by patient count
    n_best = 150
    pat_for_item = labs_df.groupby("label")["subject_id"].nunique()
    frequent_labels = pat_for_item.sort_values(ascending=False)[:n_best]
    logger.info(f"Selected top {n_best} lab tests from {len(pat_for_item)} total tests")

    # Filter to only include the most frequent lab tests
    labs_filtered = labs_df.loc[labs_df["label"].isin(frequent_labels.index)]

    logger.info(f"After filtering for top lab tests: {len(labs_filtered)} rows")
    logger.info(
        f"Number of unique patients after final filtering: {labs_filtered['subject_id'].nunique()}"
    )

    return labs_filtered


def get_output_events(db: BaseDb, output_db: BaseDb) -> pd.DataFrame:
    # Get admission IDs from the processed database
    adm_ids_df = output_db.read("SELECT hadm_id FROM admissions")
    logger.info(f"Filtering output events for {len(adm_ids_df)} admissions")

    # Create a query that joins outputevents with d_items and filters by admission IDs
    full_query = """
    SELECT 
        oe.subject_id, 
        oe.hadm_id, 
        oe.charttime, 
        oe.itemid, 
        oe.value,
        oe.valueuom,
        di.label
    FROM 
        mimiciv_icu.outputevents oe
    JOIN
        mimiciv_icu.d_items di
        ON oe.itemid = di.itemid
    WHERE 
        oe.hadm_id IN (SELECT hadm_id FROM adm_ids_df)
    """

    outputs_with_labels = db.read_with_dataframes(full_query, {"adm_ids_df": adm_ids_df})

    logger.info(f"Loaded {len(outputs_with_labels)} output events")
    logger.info(f"Number of unique patients: {outputs_with_labels['subject_id'].nunique()}")

    # List of output categories to retain
    output_label_list = [
        "Foley",
        "Void",
        "OR Urine",
        "Chest Tube #1",
        "Oral Gastric",
        "Pre-Admission",
        "TF Residual",
        "OR EBL",
        "Emesis",
        "Nasogastric",
        "Stool",
        "Jackson Pratt #1",
        "Straight Cath",
        "TF Residual Output",
        "Fecal Bag",
    ]

    # Filter to keep only output categories in the retained list
    outputs_filtered = outputs_with_labels.loc[
        outputs_with_labels["label"].isin(output_label_list)
    ].copy()

    logger.info(f"After filtering for specific output categories: {len(outputs_filtered)} rows")
    logger.info(
        f"Number of unique patients after filtering: {outputs_filtered['subject_id'].nunique()}"
    )

    # Log the unit distribution to verify consistency
    unit_counts = outputs_filtered.groupby(["label", "valueuom"]).size()
    # assert all valuoms are in mL
    if not all(unit_counts.index.get_level_values(1) == "mL"):
        logger.warning(
            f"Found non-mL units in outputs: {unit_counts[unit_counts.index.get_level_values(1) != 'mL']}"
        )
    else:
        logger.info("All output events are in mL units")

    return outputs_filtered


def get_prescriptions(db: BaseDb, output_db: BaseDb) -> pd.DataFrame:
    # Get admission IDs from the processed database
    adm_ids_df = output_db.read("SELECT hadm_id FROM admissions")
    logger.info(f"Filtering prescriptions for {len(adm_ids_df)} admissions")

    # Create a query that joins prescriptions with d_items and filters by admission IDs
    full_query = """
    SELECT 
       *
    FROM 
        mimiciv_hosp.prescriptions p
    WHERE 
        p.hadm_id IN (SELECT hadm_id FROM adm_ids_df)
    """

    prescriptions_df = db.read_with_dataframes(full_query, {"adm_ids_df": adm_ids_df})

    logger.info(f"Loaded {len(prescriptions_df)} prescriptions")
    logger.info(f"Number of unique patients: {prescriptions_df['subject_id'].nunique()}")

    # Select specific drugs from the medication list see https://github.com/jingge326/ivpvae/blob/main/preprocess/pre_mimic4/prescriptions.ipynb
    drugs_list = [
        "Acetaminophen",
        "Aspirin",
        "Bisacodyl",
        "Insulin",
        "Heparin",
        "Docusate Sodium",
        "D5W",
        "Potassium Chloride",
        "Magnesium Sulfate",
        "Metoprolol Tartrate",
        "Sodium Chloride 0.9% Flush",
        "Pantoprazole",
    ]
    prescriptions_df = prescriptions_df.loc[prescriptions_df["drug"].isin(drugs_list)]

    logger.info(f"Filtered to {len(drugs_list)} specific drugs")
    logger.info(
        f"Number of unique patients after drug filtering: {prescriptions_df['subject_id'].nunique()}"
    )

    # Standardize units and remove entries with non-standard units
    # First drop rows with null dose units
    prescriptions_df = prescriptions_df.dropna(subset=["dose_unit_rx"])

    # Standardize units and filter by expected units for each drug
    # Correct ml -> mL for specific drugs
    prescriptions_df.loc[
        (prescriptions_df["drug"] == "D5W") & (prescriptions_df["dose_unit_rx"] == "ml"),
        "dose_unit_rx",
    ] = "mL"

    prescriptions_df.loc[
        (prescriptions_df["drug"] == "Sodium Chloride 0.9% Flush")
        & (prescriptions_df["dose_unit_rx"] == "ml"),
        "dose_unit_rx",
    ] = "mL"

    # Define expected units for each drug
    drug_unit_requirements = {
        "Acetaminophen": "mg",
        "D5W": "mL",
        "Heparin": "UNIT",
        "Insulin": "UNIT",
        "Magnesium Sulfate": "gm",
        "Potassium Chloride": "mEq",
        "Bisacodyl": "mg",
        "Pantoprazole": "mg",
        "Docusate Sodium": "mg",
        "Metoprolol Tartrate": "mg",
    }

    # Filter out rows that don't match expected units for each drug
    for drug, required_unit in drug_unit_requirements.items():
        before_count = len(prescriptions_df)
        mask = ~(
            (prescriptions_df["drug"] == drug) & (prescriptions_df["dose_unit_rx"] != required_unit)
        )
        prescriptions_df = prescriptions_df.loc[mask]
        removed = before_count - len(prescriptions_df)
        if removed > 0:
            logger.info(
                f"Removed {removed} {drug} records with incorrect unit (keeping only {required_unit})"
            )

    # Clean and transform dose values to numeric format
    pre_numeric_num_entries = len(prescriptions_df)
    # Remove entries with missing dose values
    prescriptions_df.dropna(subset=["dose_val_rx"], inplace=True)

    # Remove entries with apostrophes in dose values
    prescriptions_df = prescriptions_df.loc[
        ~prescriptions_df["dose_val_rx"].str.contains("'")
    ].copy()

    # Process ranges (e.g., "10-20") by taking the mean
    range_df = prescriptions_df.loc[prescriptions_df["dose_val_rx"].str.contains("-")].copy()
    if len(range_df) > 0:
        range_df["first_digit"] = range_df["dose_val_rx"].str.split("-").str[0]
        range_df.loc[range_df["first_digit"] == "", "first_digit"] = "0.0"
        range_df["first_digit"] = range_df["first_digit"].astype(float)

        range_df["second_digit"] = range_df["dose_val_rx"].str.split("-").str[1]
        range_df.loc[range_df["second_digit"] == "", "second_digit"] = range_df.loc[
            range_df["second_digit"] == "", "first_digit"
        ]
        range_df["second_digit"] = range_df["second_digit"].astype(float)

        range_df["dose_val_rx"] = (range_df["first_digit"] + range_df["second_digit"]) / 2
        range_df.drop(columns=["first_digit", "second_digit"], inplace=True)

    # Process non-range values separately
    non_range_df = prescriptions_df.loc[~prescriptions_df["dose_val_rx"].str.contains("-")].copy()
    non_range_df["dose_val_rx"] = pd.to_numeric(non_range_df["dose_val_rx"], errors="coerce")
    non_range_df.dropna(subset=["dose_val_rx"], inplace=True)

    # Combine both dataframes
    prescriptions_df = pd.concat([non_range_df, range_df])

    entries_lost = pre_numeric_num_entries - len(prescriptions_df)
    logger.info(
        f"Dose value processing complete. Lost {entries_lost} entries ({entries_lost / pre_numeric_num_entries:.1%}) due to invalid dose values"
    )
    prescriptions_df["charttime"] = pd.to_datetime(prescriptions_df["starttime"])
    prescriptions_df["drug"] = prescriptions_df["drug"] + "_drug"

    return prescriptions_df


def get_antibiotics(db: BaseDb, output_db: BaseDb) -> pd.DataFrame:
    """
    This function implements a derived antibiotics table based on the MIMIC-IV concept SQL query.
    See: https://github.com/MIT-LCP/mimic-code/blob/c34baed99d326d438f7b9a74eea68463925063dd/mimic-iv/concepts/medication/antibiotic.sql#L169
    """

    abx_query = """
        WITH abx AS (
            SELECT DISTINCT
                drug
                , route
                , CASE
                    WHEN LOWER(drug) LIKE '%adoxa%' THEN 1
                    WHEN LOWER(drug) LIKE '%ala-tet%' THEN 1
                    WHEN LOWER(drug) LIKE '%alodox%' THEN 1
                    WHEN LOWER(drug) LIKE '%amikacin%' THEN 1
                    WHEN LOWER(drug) LIKE '%amikin%' THEN 1
                    WHEN LOWER(drug) LIKE '%amoxicill%' THEN 1
                    WHEN LOWER(drug) LIKE '%amphotericin%' THEN 1
                    WHEN LOWER(drug) LIKE '%anidulafungin%' THEN 1
                    WHEN LOWER(drug) LIKE '%ancef%' THEN 1
                    WHEN LOWER(drug) LIKE '%clavulanate%' THEN 1
                    WHEN LOWER(drug) LIKE '%ampicillin%' THEN 1
                    WHEN LOWER(drug) LIKE '%augmentin%' THEN 1
                    WHEN LOWER(drug) LIKE '%avelox%' THEN 1
                    WHEN LOWER(drug) LIKE '%avidoxy%' THEN 1
                    WHEN LOWER(drug) LIKE '%azactam%' THEN 1
                    WHEN LOWER(drug) LIKE '%azithromycin%' THEN 1
                    WHEN LOWER(drug) LIKE '%aztreonam%' THEN 1
                    WHEN LOWER(drug) LIKE '%axetil%' THEN 1
                    WHEN LOWER(drug) LIKE '%bactocill%' THEN 1
                    WHEN LOWER(drug) LIKE '%bactrim%' THEN 1
                    WHEN LOWER(drug) LIKE '%bactroban%' THEN 1
                    WHEN LOWER(drug) LIKE '%bethkis%' THEN 1
                    WHEN LOWER(drug) LIKE '%biaxin%' THEN 1
                    WHEN LOWER(drug) LIKE '%bicillin l-a%' THEN 1
                    WHEN LOWER(drug) LIKE '%cayston%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefazolin%' THEN 1
                    WHEN LOWER(drug) LIKE '%cedax%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefoxitin%' THEN 1
                    WHEN LOWER(drug) LIKE '%ceftazidime%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefaclor%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefadroxil%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefdinir%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefditoren%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefepime%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefotan%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefotetan%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefotaxime%' THEN 1
                    WHEN LOWER(drug) LIKE '%ceftaroline%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefpodoxime%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefpirome%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefprozil%' THEN 1
                    WHEN LOWER(drug) LIKE '%ceftibuten%' THEN 1
                    WHEN LOWER(drug) LIKE '%ceftin%' THEN 1
                    WHEN LOWER(drug) LIKE '%ceftriaxone%' THEN 1
                    WHEN LOWER(drug) LIKE '%cefuroxime%' THEN 1
                    WHEN LOWER(drug) LIKE '%cephalexin%' THEN 1
                    WHEN LOWER(drug) LIKE '%cephalothin%' THEN 1
                    WHEN LOWER(drug) LIKE '%cephapririn%' THEN 1
                    WHEN LOWER(drug) LIKE '%chloramphenicol%' THEN 1
                    WHEN LOWER(drug) LIKE '%cipro%' THEN 1
                    WHEN LOWER(drug) LIKE '%ciprofloxacin%' THEN 1
                    WHEN LOWER(drug) LIKE '%claforan%' THEN 1
                    WHEN LOWER(drug) LIKE '%clarithromycin%' THEN 1
                    WHEN LOWER(drug) LIKE '%cleocin%' THEN 1
                    WHEN LOWER(drug) LIKE '%clindamycin%' THEN 1
                    WHEN LOWER(drug) LIKE '%cubicin%' THEN 1
                    WHEN LOWER(drug) LIKE '%dicloxacillin%' THEN 1
                    WHEN LOWER(drug) LIKE '%dirithromycin%' THEN 1
                    WHEN LOWER(drug) LIKE '%doryx%' THEN 1
                    WHEN LOWER(drug) LIKE '%doxycy%' THEN 1
                    WHEN LOWER(drug) LIKE '%duricef%' THEN 1
                    WHEN LOWER(drug) LIKE '%dynacin%' THEN 1
                    WHEN LOWER(drug) LIKE '%ery-tab%' THEN 1
                    WHEN LOWER(drug) LIKE '%eryped%' THEN 1
                    WHEN LOWER(drug) LIKE '%eryc%' THEN 1
                    WHEN LOWER(drug) LIKE '%erythrocin%' THEN 1
                    WHEN LOWER(drug) LIKE '%erythromycin%' THEN 1
                    WHEN LOWER(drug) LIKE '%factive%' THEN 1
                    WHEN LOWER(drug) LIKE '%flagyl%' THEN 1
                    WHEN LOWER(drug) LIKE '%fortaz%' THEN 1
                    WHEN LOWER(drug) LIKE '%furadantin%' THEN 1
                    WHEN LOWER(drug) LIKE '%garamycin%' THEN 1
                    WHEN LOWER(drug) LIKE '%gentamicin%' THEN 1
                    WHEN LOWER(drug) LIKE '%kanamycin%' THEN 1
                    WHEN LOWER(drug) LIKE '%keflex%' THEN 1
                    WHEN LOWER(drug) LIKE '%kefzol%' THEN 1
                    WHEN LOWER(drug) LIKE '%ketek%' THEN 1
                    WHEN LOWER(drug) LIKE '%levaquin%' THEN 1
                    WHEN LOWER(drug) LIKE '%levofloxacin%' THEN 1
                    WHEN LOWER(drug) LIKE '%lincocin%' THEN 1
                    WHEN LOWER(drug) LIKE '%linezolid%' THEN 1
                    WHEN LOWER(drug) LIKE '%macrobid%' THEN 1
                    WHEN LOWER(drug) LIKE '%macrodantin%' THEN 1
                    WHEN LOWER(drug) LIKE '%maxipime%' THEN 1
                    WHEN LOWER(drug) LIKE '%mefoxin%' THEN 1
                    WHEN LOWER(drug) LIKE '%metronidazole%' THEN 1
                    WHEN LOWER(drug) LIKE '%meropenem%' THEN 1
                    WHEN LOWER(drug) LIKE '%methicillin%' THEN 1
                    WHEN LOWER(drug) LIKE '%minocin%' THEN 1
                    WHEN LOWER(drug) LIKE '%minocycline%' THEN 1
                    WHEN LOWER(drug) LIKE '%monodox%' THEN 1
                    WHEN LOWER(drug) LIKE '%monurol%' THEN 1
                    WHEN LOWER(drug) LIKE '%morgidox%' THEN 1
                    WHEN LOWER(drug) LIKE '%moxatag%' THEN 1
                    WHEN LOWER(drug) LIKE '%moxifloxacin%' THEN 1
                    WHEN LOWER(drug) LIKE '%mupirocin%' THEN 1
                    WHEN LOWER(drug) LIKE '%myrac%' THEN 1
                    WHEN LOWER(drug) LIKE '%nafcillin%' THEN 1
                    WHEN LOWER(drug) LIKE '%neomycin%' THEN 1
                    WHEN LOWER(drug) LIKE '%nicazel doxy 30%' THEN 1
                    WHEN LOWER(drug) LIKE '%nitrofurantoin%' THEN 1
                    WHEN LOWER(drug) LIKE '%norfloxacin%' THEN 1
                    WHEN LOWER(drug) LIKE '%noroxin%' THEN 1
                    WHEN LOWER(drug) LIKE '%ocudox%' THEN 1
                    WHEN LOWER(drug) LIKE '%ofloxacin%' THEN 1
                    WHEN LOWER(drug) LIKE '%omnicef%' THEN 1
                    WHEN LOWER(drug) LIKE '%oracea%' THEN 1
                    WHEN LOWER(drug) LIKE '%oraxyl%' THEN 1
                    WHEN LOWER(drug) LIKE '%oxacillin%' THEN 1
                    WHEN LOWER(drug) LIKE '%pc pen vk%' THEN 1
                    WHEN LOWER(drug) LIKE '%pce dispertab%' THEN 1
                    WHEN LOWER(drug) LIKE '%panixine%' THEN 1
                    WHEN LOWER(drug) LIKE '%pediazole%' THEN 1
                    WHEN LOWER(drug) LIKE '%penicillin%' THEN 1
                    WHEN LOWER(drug) LIKE '%periostat%' THEN 1
                    WHEN LOWER(drug) LIKE '%pfizerpen%' THEN 1
                    WHEN LOWER(drug) LIKE '%piperacillin%' THEN 1
                    WHEN LOWER(drug) LIKE '%tazobactam%' THEN 1
                    WHEN LOWER(drug) LIKE '%primsol%' THEN 1
                    WHEN LOWER(drug) LIKE '%proquin%' THEN 1
                    WHEN LOWER(drug) LIKE '%raniclor%' THEN 1
                    WHEN LOWER(drug) LIKE '%rifadin%' THEN 1
                    WHEN LOWER(drug) LIKE '%rifampin%' THEN 1
                    WHEN LOWER(drug) LIKE '%rocephin%' THEN 1
                    WHEN LOWER(drug) LIKE '%smz-tmp%' THEN 1
                    WHEN LOWER(drug) LIKE '%septra%' THEN 1
                    WHEN LOWER(drug) LIKE '%septra ds%' THEN 1
                    WHEN LOWER(drug) LIKE '%septra%' THEN 1
                    WHEN LOWER(drug) LIKE '%solodyn%' THEN 1
                    WHEN LOWER(drug) LIKE '%spectracef%' THEN 1
                    WHEN LOWER(drug) LIKE '%streptomycin%' THEN 1
                    WHEN LOWER(drug) LIKE '%sulfadiazine%' THEN 1
                    WHEN LOWER(drug) LIKE '%sulfamethoxazole%' THEN 1
                    WHEN LOWER(drug) LIKE '%trimethoprim%' THEN 1
                    WHEN LOWER(drug) LIKE '%sulfatrim%' THEN 1
                    WHEN LOWER(drug) LIKE '%sulfisoxazole%' THEN 1
                    WHEN LOWER(drug) LIKE '%suprax%' THEN 1
                    WHEN LOWER(drug) LIKE '%synercid%' THEN 1
                    WHEN LOWER(drug) LIKE '%tazicef%' THEN 1
                    WHEN LOWER(drug) LIKE '%tetracycline%' THEN 1
                    WHEN LOWER(drug) LIKE '%timentin%' THEN 1
                    WHEN LOWER(drug) LIKE '%tobramycin%' THEN 1
                    WHEN LOWER(drug) LIKE '%trimethoprim%' THEN 1
                    WHEN LOWER(drug) LIKE '%unasyn%' THEN 1
                    WHEN LOWER(drug) LIKE '%vancocin%' THEN 1
                    WHEN LOWER(drug) LIKE '%vancomycin%' THEN 1
                    WHEN LOWER(drug) LIKE '%vantin%' THEN 1
                    WHEN LOWER(drug) LIKE '%vibativ%' THEN 1
                    WHEN LOWER(drug) LIKE '%vibra-tabs%' THEN 1
                    WHEN LOWER(drug) LIKE '%vibramycin%' THEN 1
                    WHEN LOWER(drug) LIKE '%zinacef%' THEN 1
                    WHEN LOWER(drug) LIKE '%zithromax%' THEN 1
                    WHEN LOWER(drug) LIKE '%zosyn%' THEN 1
                    WHEN LOWER(drug) LIKE '%zyvox%' THEN 1
                    ELSE 0
                END AS antibiotic
            FROM mimiciv_hosp.prescriptions
            -- excludes vials/syringe/normal saline, etc
            WHERE drug_type NOT IN ('BASE')
                -- we exclude routes via the eye, ears, or topically
                AND route NOT IN ('OU', 'OS', 'OD', 'AU', 'AS', 'AD', 'TP')
                AND LOWER(route) NOT LIKE '%ear%'
                AND LOWER(route) NOT LIKE '%eye%'
                -- we exclude certain types of antibiotics: topical creams,
                -- gels, desens, etc
                AND LOWER(drug) NOT LIKE '%cream%'
                AND LOWER(drug) NOT LIKE '%desensitization%'
                AND LOWER(drug) NOT LIKE '%ophth oint%'
                AND LOWER(drug) NOT LIKE '%gel%'
        -- other routes not sure about...
        -- for sure keep: ('IV','PO','PO/NG','ORAL', 'IV DRIP', 'IV BOLUS')
        -- ? VT, PB, PR, PL, NS, NG, NEB, NAS, LOCK, J TUBE, IVT
        -- ? IT, IRR, IP, IO, INHALATION, IN, IM
        -- ? IJ, IH, G TUBE, DIALYS
        -- ?? enemas??
        )

        SELECT
            pr.subject_id, pr.hadm_id
            , ie.stay_id
            , pr.drug AS antibiotic
            , pr.route
            , pr.starttime
            , pr.stoptime
        FROM mimiciv_hosp.prescriptions pr
        -- inner join to subselect to only antibiotic prescriptions
        INNER JOIN abx
            ON pr.drug = abx.drug
                -- route is never NULL for antibiotics
                -- only ~4000 null rows in prescriptions total.
                AND pr.route = abx.route
        -- add in stay_id as we use this table for sepsis-3
        LEFT JOIN mimiciv_icu.icustays ie
            ON pr.hadm_id = ie.hadm_id
                AND pr.starttime >= ie.intime
                AND pr.starttime < ie.outtime
        WHERE abx.antibiotic = 1
            AND pr.hadm_id IN (SELECT hadm_id FROM adm_ids_df)     
        
        ;
    """

    adm_ids_df = output_db.read("SELECT hadm_id FROM admissions")
    logger.info(f"Filtering antibiotics for {len(adm_ids_df)} admissions")

    abx_df = db.read_with_dataframes(abx_query, {"adm_ids_df": adm_ids_df})

    return abx_df
