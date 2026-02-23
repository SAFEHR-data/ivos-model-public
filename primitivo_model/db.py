import shutil
from abc import ABC
from datetime import datetime
from typing import Any, Dict, Optional

import duckdb
import pandas as pd

from primitivo_model.config import settings


class BaseDb(ABC):
    def __init__(self, name: str, level: str, read_only: bool = True, back_up=False):
        self._name = name
        self._level = level
        self._read_only = read_only
        self._back_up = back_up

        self.path = settings.DATA_ROOT / self._name / f"{self._name}{self._level}.db"
        if self._read_only and not self.exists():
            raise FileNotFoundError(f"{self} not found")

        (settings.DATA_ROOT / self._name).mkdir(exist_ok=True, parents=True)
        if not self._read_only:
            self.reset()

    def exists(self) -> bool:
        return self.path.exists()

    def reset(self) -> None:
        if self.exists() and self._back_up:
            dt = datetime.now().isoformat(timespec="minutes").replace(":", "-")
            bak_path = settings.DATA_ROOT / self._name / f"{self._name}{self._level}_{dt}.db"
            shutil.copy(self.path, bak_path)

    def read(self, query: str, dtypes: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        with duckdb.connect(self.path, read_only=True) as conn:
            df = conn.execute(query).df()

        if dtypes:
            df = df.astype(dtype=dtypes)

        return df

    def read_with_dataframes(
        self,
        query: str,
        dataframes: Dict[str, pd.DataFrame],
        dtypes: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        with duckdb.connect(self.path, read_only=True) as conn:
            # Register each DataFrame
            for name, df in dataframes.items():
                conn.register(name, df)

            # Execute the query
            df = conn.execute(query).df()

        if dtypes:
            df = df.astype(dtype=dtypes)

        return df

    def save(self, table_name: str, df: pd.DataFrame) -> None:
        with duckdb.connect(self.path) as conn:
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")

    def __str__(self) -> str:
        return f"BaseDb(name={self._name}, level={self._level}, read_only={self._read_only}, path='{self.path}')"
