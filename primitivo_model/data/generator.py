import pickle

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

import lab.torch as B
from primitivo_model.config import settings
from primitivo_model.data.sources import DataSource, create_data_source
from primitivo_model.data.tasks import (
    ForecastSplitter,
    TaskSplitter,
)


class TaskSet:
    def __init__(
        self,
        split_strategy: TaskSplitter,  # Allow custom strategy to be passed
        data_source: DataSource,
        task_length_hours=None,
        min_task_measurements=10,
        subset="train",
        smoke_test=False,
        refresh_cache=False,
        **kwargs,  # Allow subclasses to pass additional parameters
    ):
        self.dtype = np.float32

        # Create the random state on the right device.
        # hardcode the seed and ensure distinct seeds are used for each split
        seed = 101
        seed += 1 if subset == "val" else 0
        seed += 2 if subset == "test" else 0

        self.pre_process_dtype = np.float32
        # this is used when deciding forecast horizons for the tasks
        with B.on_device("cpu"):
            self.state = B.create_random_state(self.dtype, seed)

        self.split_strategy = split_strategy
        self.task_length_hours = task_length_hours
        self.min_task_measurements = min_task_measurements
        self.subset = subset
        self.smoke_test = smoke_test
        self.refresh_cache = refresh_cache

        # Create appropriate data source
        self.data_source = data_source
        self.measurements_df, self.std_params = self.data_source.get_subset(subset)

        if smoke_test:
            # only use the first 10 encounters
            logger.warning(f"Running smoke test, using only 30 encounters for {subset} data")
            smoke_test_ids = self.measurements_df.pat_enc_csn_id.unique()[:30]
            self.measurements_df = self.measurements_df[
                self.measurements_df.pat_enc_csn_id.isin(smoke_test_ids)
            ]

        self.measurement_names = self.get_measurement_names()
        self.num_measurements = len(self.measurement_names)

        # Handle cache refresh if requested
        if self.refresh_cache and not self.smoke_test:
            self.clear_cache()

        # Try to load from cache first, otherwise compute tasks
        cached_data = self.load_tasks()
        if cached_data is not None:
            self.tasks = cached_data["tasks"]
            self.task_windows_df = cached_data["task_windows_df"]
        else:
            self.tasks = self._get_tasks()
            # Save to cache for next time (task_windows_df is created in _get_tasks)
            if not smoke_test:
                self.save_tasks()

        self.encounter_ids = list(self.tasks.keys())
        self._log_task_statistics()

    def __len__(self):
        return len(self.encounter_ids)

    def __getitem__(self, enc_id):
        return self.tasks[enc_id]

    def get_min_timestep(self):
        return (
            self.measurements_df.sort_values("enc_elapsed_time")
            .groupby(["pat_enc_csn_id", "name"])["enc_elapsed_time"]
            .diff()
            .min()
        )

    def get_measurement_names(self):
        return sorted(
            list(self.data_source.dense_measurements | self.data_source.sparse_measurements)
        )

    def generate_task_windows(self, enc_id, max_enc_time):
        """Generate task windows for an encounter. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_task_windows")

    def _generate_all_task_windows(self):
        """Generate a DataFrame of all task windows across all encounters."""
        all_windows = []

        # Group encounters by pat_enc_csn_id
        for enc_id, enc_df in tqdm(
            self.measurements_df.groupby("pat_enc_csn_id"), desc="Generating task windows"
        ):
            max_enc_time = enc_df.enc_elapsed_time.max()

            # Generate task windows using subclass-specific strategy
            task_windows = self.generate_task_windows(enc_id, max_enc_time)

            for task_id, (start_time, prediction_time, end_time) in task_windows:
                all_windows.append(
                    {
                        "task_id": task_id,
                        "enc_id": str(enc_id),
                        "start_time": start_time,
                        "prediction_time": prediction_time,
                        "end_time": end_time,
                    }
                )

        if not all_windows:
            return pd.DataFrame(
                columns=["task_id", "enc_id", "start_time", "prediction_time", "end_time"]
            )

        windows_df = pd.DataFrame(all_windows)
        windows_df = windows_df.set_index("task_id")

        return windows_df

    def _get_tasks(self):
        # Generate all task windows as a DataFrame
        self.task_windows_df = self._generate_all_task_windows()

        if len(self.task_windows_df) == 0:
            logger.warning("No task windows generated")
            return {}

        logger.info(f"Generated {len(self.task_windows_df)} task windows")

        all_tasks = {}

        # Pre-group measurements by encounter for efficient lookup
        enc_dfs = {
            str(enc_id): enc_df.sort_values("enc_elapsed_time")
            for enc_id, enc_df in self.measurements_df.groupby("pat_enc_csn_id")
        }

        # Group windows by encounter for efficient processing
        for enc_id, enc_windows in tqdm(
            self.task_windows_df.groupby("enc_id"), desc="Processing tasks"
        ):
            enc_id_str = str(enc_id)
            enc_df_sorted = enc_dfs[enc_id_str]
            enc_times = enc_df_sorted.enc_elapsed_time.values

            for task_id, window in enc_windows.iterrows():
                start_time = window["start_time"]
                prediction_time = window["prediction_time"]
                end_time = window["end_time"]

                # Use binary search for efficient time-based filtering
                start_idx = enc_times.searchsorted(start_time, side="left")
                end_idx = enc_times.searchsorted(end_time, side="left")
                pred_idx = enc_times.searchsorted(prediction_time, side="left")

                # minimum number of context points
                if pred_idx - start_idx < self.min_task_measurements:
                    continue

                # no target points
                if pred_idx == end_idx:
                    continue

                # Slice the sorted dataframe efficiently
                task_df = enc_df_sorted.iloc[start_idx:end_idx]

                # Split the task into contexts and targets
                enc_ctx, enc_xt, enc_yt = self.split_strategy.split_task(
                    task_df, self.measurement_names, prediction_time, self.pre_process_dtype
                )

                all_tasks[task_id] = {"contexts": enc_ctx, "xt": enc_xt, "yt": enc_yt}

        return all_tasks

    def _log_task_statistics(self):
        """Log statistics about the tasks in this set"""
        # Count empty vs non-empty context and target sets
        empty_ctx_counts = 0
        empty_tgt_counts = 0
        total_measurements = 0

        for task_id, task in self.tasks.items():
            for i, ((ctx_x, ctx_y), tgt_x) in enumerate(zip(task["contexts"], task["xt"])):
                total_measurements += 1
                if ctx_x.shape[-1] == 0:
                    empty_ctx_counts += 1
                if tgt_x.shape[-1] == 0:
                    empty_tgt_counts += 1

        logger.info(f"TaskSet {self.__class__.__name__} statistics:")
        logger.info(f"  - Total tasks: {len(self.tasks)}")
        logger.info(f"  - Total measurements: {total_measurements}")
        logger.info(
            f"  - Empty context sets: {empty_ctx_counts} ({empty_ctx_counts / total_measurements * 100:.1f}%)"
        )
        logger.info(
            f"  - Empty target sets: {empty_tgt_counts} ({empty_tgt_counts / total_measurements * 100:.1f}%)"
        )

    def get_cache_filename(self):
        """Generate a filename for caching this task set. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement get_cache_filename")

    def get_cache_path(self):
        """Get the full path for caching this task set."""
        cache_dir = settings.DATA_ROOT / "tasks"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / self.get_cache_filename()

    def save_tasks(self):
        """Save tasks and task windows DataFrame to cache."""
        if self.smoke_test:
            logger.info("Skipping cache save for smoke test")
            return

        cache_path = self.get_cache_path()
        logger.info(f"Saving {len(self.tasks)} tasks to {cache_path}")

        try:
            cache_data = {
                "tasks": self.tasks,
                "task_windows_df": self.task_windows_df,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info("Successfully saved tasks and task windows to cache")
        except Exception as e:
            logger.error(f"Failed to save tasks to cache: {e}")

    def load_tasks(self):
        """Load tasks and task windows DataFrame from cache if available."""
        if self.smoke_test:
            return None

        cache_path = self.get_cache_path()

        if not cache_path.exists():
            logger.debug(f"No cache found at {cache_path}")
            return None

        try:
            logger.info(f"Loading tasks from {cache_path}")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            logger.info(
                f"Successfully loaded {len(cached_data['tasks'])} tasks and task windows from cache"
            )
            return cached_data
        except Exception as e:
            logger.warning(f"Failed to load tasks from cache: {e}")
            return None

    def clear_cache(self):
        """Clear the cache file for this task set."""
        cache_path = self.get_cache_path()
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cleared cache file: {cache_path}")
        else:
            logger.info("No cache file to clear")


class SampledForecastTaskSet(TaskSet):
    def __init__(
        self,
        forecast_hours=12.0,
        lookback_hours=48.0,
        sparse_only=False,  # or "all"
        min_task_measurements=10,
        subset="train",
        route="mock",
        smoke_test=False,
        data_source="radix",
        tasks_per_day=1.0,
        refresh_cache=False,
    ):
        data_source = create_data_source(data_source, route)
        split_strategy = ForecastSplitter(
            forecast_hours=forecast_hours,
            sparse_only=sparse_only,
            sparse_measurements=data_source.sparse_measurements,
        )

        task_length_hours = lookback_hours + forecast_hours
        self.forecast_hours = forecast_hours
        self.lookback_hours = lookback_hours
        self.tasks_per_day = tasks_per_day
        self.sparse_only = sparse_only

        super().__init__(
            split_strategy,
            data_source,
            task_length_hours,
            min_task_measurements,
            subset,
            smoke_test,
            refresh_cache,
        )

    def generate_task_windows(self, enc_id, max_enc_time):
        """Generate randomly sampled task windows"""
        task_windows = []

        # If the encounter is shorter than the task length we skip
        if max_enc_time < self.task_length_hours:
            return task_windows

        # Calculate how many tasks fit into the encounter
        num_tasks = (max_enc_time / 24) * self.tasks_per_day

        # this is the decimal part of num_tasks
        remaining_tasks = num_tasks % 1

        # Decide whether to create a remainder task
        create_remainder = False
        if remaining_tasks > 0:
            self.state, remainder_cutoff = B.random.rand(self.state, self.pre_process_dtype, 1)
            create_remainder = remainder_cutoff < remaining_tasks

        num_full_tasks = int(num_tasks)
        total_tasks = num_full_tasks + (1 if create_remainder else 0)

        # Generate random start times for each task
        for i in range(total_tasks):
            # sample uniformly from valid start positions
            max_start_time = max_enc_time - self.task_length_hours
            self.state, start_time = B.random.rand(self.state, self.pre_process_dtype, 1)
            start_time = start_time[0] * max_start_time
            prediction_time = start_time + self.lookback_hours
            end_time = start_time + self.task_length_hours

            task_id = f"{enc_id}n{i}"
            task_windows.append((task_id, (start_time, prediction_time, end_time)))

        return task_windows

    def get_cache_filename(self):
        """Generate cache filename for SampledForecastTaskSet."""
        data_source = getattr(self.data_source, "name", "unknown")
        route = getattr(self.data_source, "route", "unknown")
        sparse_only_str = "sparse" if self.sparse_only else "all"
        return f"{data_source}_{route}_SampledForecastTaskSet_{self.subset}_{self.forecast_hours}f_{self.lookback_hours}h_{self.tasks_per_day}tpd_{sparse_only_str}_{self.min_task_measurements}min.pkl"
