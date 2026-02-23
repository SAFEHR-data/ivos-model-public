import pandas as pd
from loguru import logger

from primitivo_model.data.splitting import get_split_and_standardise
from primitivo_model.db import BaseDb


class DataSource:
    """
    Base class for data sources.
    """

    def __init__(self, name=None):
        self.name = name
        self.test_cutoff = None

    @property
    def dense_measurements(self):
        return {}

    @property
    def sparse_measurements(self):
        return {}

    def get_measurements_df(self):
        """Get measurements dataframe from the data source."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_adm_df(self):
        """Get IV abx admissions dataframe from the data source."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_abx_rx_df(self):
        """Get abx prescriptions dataframe from the data source."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_measurements_with_adm_df(self):
        measurements_df = self.get_measurements_df()
        adm_df = self.get_adm_df()
        return measurements_df.join(
            adm_df.set_index("pat_enc_csn_id").admittime, on="pat_enc_csn_id"
        )

    @staticmethod
    def normalise_dtypes(df):
        return df.astype({"pat_enc_csn_id": "string"})

    def get_subset(self, subset):
        measurements_adm_df = self.get_measurements_with_adm_df()

        # pass mimic-specific cutoff into the splitting logic
        measurements_subset, std_params = get_split_and_standardise(
            measurements_adm_df, subset, cutoff_date=self.test_cutoff
        )
        measurements_subset = measurements_subset.drop(columns=["admittime"], errors=True)
        return measurements_subset, std_params


class MimicChartsDataSource(DataSource):
    """
    Data source for mimic charts.
    """

    def __init__(self, route):
        super().__init__(name=f"mimic4-{route}")
        self.route = route
        self.test_cutoff = pd.Timestamp("2019-01-01")

    @property
    def dense_measurements(self):
        return {
            "pulse",
            "resp_rate",
            "spo2",
            "temp",
            "systolic_bp",
        }

    @property
    def sparse_measurements(self):
        base_route = self.route.strip("-dev")
        if base_route == "simple-charts":
            return set()
        elif base_route == "simple-charts-labs":
            return {"whitecell"}
        else:
            raise ValueError(f"Route {self.route} not valid")

    def get_measurements_with_adm_df(self):
        measurements_df = self.get_measurements_df()
        adm_df = self.get_adm_df()
        return measurements_df.join(
            adm_df.set_index("pat_enc_csn_id").min_real_admittime.rename("admittime"),
            on="pat_enc_csn_id",
        )

    def get_measurements_df(self) -> pd.DataFrame:
        db = BaseDb("mimic4", self.route, read_only=True)
        logger.info(f"Loading measurements from {db.path}")

        # Load all measurements
        measurements_df = db.read("SELECT * FROM measurements")

        # Map column names to match expected format
        measurements_df = measurements_df.rename(
            columns={
                "admission_id": "pat_enc_csn_id",
                "time": "enc_elapsed_time",
                "value": "value",
                "label": "name",
            }
        )

        measurements_df = measurements_df.sort_values(
            ["pat_enc_csn_id", "name", "enc_elapsed_time"]
        )

        return self.normalise_dtypes(measurements_df)

    def get_adm_df(self) -> pd.DataFrame:
        db = BaseDb("mimic4", self.route, read_only=True)
        logger.info(f"Loading adm from {db.path}")

        # Load all measurements
        adm_df = db.read("SELECT * FROM iv_abx_adm")

        # Map column names to match expected format
        adm_df = adm_df.rename(
            columns={
                "hadm_id": "pat_enc_csn_id",
            }
        )

        adm_df = adm_df.sort_values(["pat_enc_csn_id", "admittime"])

        return self.normalise_dtypes(adm_df)

    def get_abx_rx_df(self) -> pd.DataFrame:
        db = BaseDb("mimic4", self.route, read_only=True)
        logger.info(f"Loading abx rx from {db.path}")

        # Load all antibiotic prescriptions
        abx_rx_df = db.read("SELECT * FROM abx_rx")

        # Map column names to match expected format
        abx_rx_df = abx_rx_df.rename(
            columns={
                "hadm_id": "pat_enc_csn_id",
            }
        ).drop(columns=["subject_id"])

        abx_rx_df = abx_rx_df.sort_values(["pat_enc_csn_id", "starttime"])

        return self.normalise_dtypes(abx_rx_df)


class RadixDataSource(DataSource):
    """
    Data source for Radix data.
    """

    def __init__(self, route):
        super().__init__(name=f"radix-{route}")
        self.route = route
        self.test_cutoff = (
            pd.Timestamp("2025-06-01", tz="Europe/London")
            if route == "mini"
            else pd.Timestamp("2024-01-01", tz="Europe/London")
        )

    @property
    def dense_measurements(self):
        return {
            "pulse",
            "resp_rate",
            "spo2",
            "temp",
            "systolic_bp",
        }

    @property
    def sparse_measurements(self):
        return set({})

    def get_measurements_df(self):
        db = BaseDb("radix", level=self.route, read_only=True)
        logger.info(f"Loading measurements from {db.path}")

        # Load all measurements
        measurements_df = db.read("select * from measurements")
        measurements_df = measurements_df.assign(
            enc_elapsed_time=measurements_df.enc_elapsed_time.dt.total_seconds().to_numpy() / 60**2
        )

        measurements_df = measurements_df.sort_values(
            ["pat_enc_csn_id", "name", "enc_elapsed_time"]
        )

        return self.normalise_dtypes(measurements_df)

    def get_adm_df(self) -> pd.DataFrame:
        db = BaseDb("radix", level=self.route, read_only=True)
        logger.info(f"Loading adm from {db.path}")

        # Load all measurements
        adm_df = db.read("SELECT * FROM cohort")

        # Map column names to match expected format
        adm_df = adm_df.rename(
            columns={
                "enc_start_dt": "admittime",
                "enc_end_dt": "dischtime",
                "death_dt": "deathtime",
                "ethnic_group": "race",
                "sex": "gender",
            }
        )

        adm_df = adm_df.sort_values(["pat_enc_csn_id", "admittime"])

        return self.normalise_dtypes(adm_df)

    def get_abx_rx_df(self) -> pd.DataFrame:
        db = BaseDb("radix", level=self.route, read_only=True)
        logger.info(f"Loading abx rx from {db.path}")

        abx_rx_df = db.read("SELECT * FROM prescriptions")

        abx_rx_df = abx_rx_df.assign(
            starttime=abx_rx_df.starttime.dt.total_seconds().to_numpy() / 60**2,
            stoptime=abx_rx_df.stoptime.dt.total_seconds().to_numpy() / 60**2,
        )

        abx_rx_df = abx_rx_df.sort_values(["pat_enc_csn_id", "starttime"])

        return self.normalise_dtypes(abx_rx_df)


# Factory function to create data source objects
def create_data_source(data_source_type, route=None):
    if data_source_type == "mimic4":
        return MimicChartsDataSource(route)
    elif data_source_type == "radix":
        return RadixDataSource(route)
    else:
        raise ValueError(f"Unknown data source type: {data_source_type}")
