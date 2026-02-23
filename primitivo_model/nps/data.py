from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from neuralprocesses import Aggregate, AggregateInput, merge_contexts

import lab.torch as B
from primitivo_model.data.generator import TaskSet
from primitivo_model.data.util import cast_task_tensors


class TaskLoader:
    def __init__(
        self,
        task_set: TaskSet,
        batch_size: int,
        device,
        seed: int = 0,
        epoch_size: Optional[int] = None,
        dataset_fraction: float = 1.0,
        no_yt=False,
    ):
        self.task_set = task_set
        self.batch_size = batch_size
        self.device = device
        self.dtype = torch.float32
        self.int64 = torch.int64  # For B.randperm
        self.dataset_fraction = dataset_fraction
        self.no_yt = no_yt

        # Create the random state for batching on the right device.
        with B.on_device(self.device):
            self.state = B.create_random_state(self.dtype, seed)

        self.batches = self._generate_all_batches()
        self.num_unique_batches = len(self.batches)

        if epoch_size:
            if epoch_size % batch_size != 0:
                raise ValueError(
                    f"epoch_size {epoch_size} must be a multiple of batch_size {batch_size}"
                )

            self.num_batches = epoch_size // batch_size
        else:
            self.num_batches = self.num_unique_batches

        logger.info(
            f"Creating task batcher with batches in epoch: {self.num_batches}, "
            f"epoch size: {epoch_size if epoch_size is not None else self.num_batches * self.batch_size}, "
            f"batch size: {self.batch_size}"
        )

        self._batch_index = 0

        # Initial shuffle of batches
        self._shuffle_batches()

    def _generate_all_batches(self) -> List[Dict[str, Any]]:
        """Pre-generate all possible batches from the data."""
        all_batches = []

        # Apply dataset fraction to limit number of encounters used
        num_total_encounters = len(self.task_set.encounter_ids)
        num_encounters = int(num_total_encounters * self.dataset_fraction)

        if num_encounters < num_total_encounters:
            logger.info(
                f"Using {num_encounters}/{num_total_encounters} encounters "
                f"(dataset_fraction={self.dataset_fraction:.2f})"
            )
        # Shuffle encounter IDs before selecting the ones to use
        self.state, perm = B.randperm(self.state, self.int64, num_total_encounters)
        shuffled_encounter_ids = [self.task_set.encounter_ids[i] for i in perm]
        # Use only the first num_encounters
        encounter_ids_to_use = shuffled_encounter_ids[:num_encounters]

        num_complete_batches = num_encounters // self.batch_size

        # Handle complete batches
        for batch_idx in range(num_complete_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_enc_ids = encounter_ids_to_use[start_idx:end_idx]

            batch = self._create_batch_from_encounters(batch_enc_ids)
            all_batches.append(batch)

        # Handle the last incomplete batch if it exists
        remaining = num_encounters % self.batch_size
        if remaining > 0 and num_encounters > 0:  # Ensure there are encounters to batch
            start_idx = num_complete_batches * self.batch_size
            # Only use remaining encounters without wrapping around to avoid duplicates
            batch_enc_ids = encounter_ids_to_use[start_idx:]

            batch = self._create_batch_from_encounters(batch_enc_ids)
            all_batches.append(batch)
        elif num_encounters == 0:
            logger.warning("No encounters found in TaskSet, no batches will be generated.")

        logger.info(f"Pre-generated {len(all_batches)} batches from {num_encounters} tasks")
        return all_batches

    def _create_batch_from_encounters(self, enc_ids):
        """Create a batch from a list of encounter IDs."""
        return create_batch_from_tasks(
            self.task_set.tasks,
            enc_ids,
            self.dtype,
            self.device,
            self.task_set.measurement_names,
            self.no_yt,
        )

    def _shuffle_batches(self):
        """Shuffle the order of batches."""
        if self.num_unique_batches > 0:
            self.state, perm = B.randperm(self.state, self.int64, self.num_unique_batches)
            self.batch_order = perm
        else:
            self.batch_order = []  # No batches to shuffle
        self._batch_index = 0

    def generate_batch(self):
        """Return the next pre-generated batch in the shuffled order."""
        if self.num_batches == 0:
            logger.warning("No batches available to generate.")
            # Depending on desired behavior, could raise an error or return None
            return None

        # Get the next batch using the shuffled order
        batch_idx = self.batch_order[self._batch_index]
        self._batch_index += 1

        if self._batch_index >= self.num_unique_batches:
            self._shuffle_batches()  # Resets _batch_index to 0

        # We've reached the end of the epoch. Shuffle for next epoch.
        elif self._batch_index >= self.num_batches:
            self._shuffle_batches()

        return self.batches[batch_idx]

    def epoch(self):
        self._batch_index = 0

        def lazy_gen_batch():
            return self.generate_batch()

        return (lazy_gen_batch() for _ in range(self.num_batches))


def create_batch_from_tasks(
    tasks, enc_ids, dtype, device, measurement_names: List[str], no_yt: bool
) -> Dict[str, Any]:
    """Create a single batch from a list of encounter IDs."""
    num_measurements = len(measurement_names)

    batch_contexts = []
    batch_xt = []
    batch_yt = []

    for enc_id in enc_ids:
        task = tasks[enc_id]
        task = cast_task_tensors(task, dtype, device)
        batch_contexts.append(task["contexts"])
        batch_xt.append(task["xt"])
        batch_yt.append(task["yt"])

    batch_contexts_agg = [
        merge_contexts(*[bc[i] for bc in batch_contexts]) for i in range(num_measurements)
    ]

    xt_aggregates = []
    yt_aggregates = []

    for i, name in enumerate(measurement_names):
        # Get max number of points for this measurement type
        batch_max_num_points = max([bx[i].shape[2] for bx in batch_xt])

        tensors_xt = [bx[i] for bx in batch_xt]

        # Get actual sizes and find which tensors need padding
        actual_sizes = [t.shape[2] for t in tensors_xt]
        need_padding = [size < batch_max_num_points for size in actual_sizes]

        # Process and pad tensors if needed
        padded_xt = []
        for j, tensor_xt in enumerate(tensors_xt):
            if need_padding[j]:
                pad_size = batch_max_num_points - actual_sizes[j]
                pad_xt = B.zeros(dtype, 1, 1, pad_size)
                padded_xt.append(B.concat(tensor_xt, pad_xt, axis=2))

            else:
                padded_xt.append(tensor_xt)

        xt_aggregates.append((B.concat(*padded_xt, axis=0), i))

        if not no_yt:
            tensors_yt = [by[i] for by in batch_yt]
            padded_yt = []
            for j, tensor_yt in enumerate(tensors_yt):
                if need_padding[j]:
                    pad_size = batch_max_num_points - actual_sizes[j]
                    pad_yt = B.ones(dtype, 1, 1, pad_size) * float("nan")
                    padded_yt.append(B.concat(tensor_yt, pad_yt, axis=2))
                else:
                    padded_yt.append(tensor_yt)
            # Concatenate all tensors
            yt_aggregates.append((B.concat(*padded_yt, axis=0)))

    batch_xt_agg = AggregateInput(*xt_aggregates)

    if not no_yt:
        batch_yt_agg = Aggregate(*yt_aggregates)
    else:
        batch_yt_agg = None

    return {"ids": enc_ids, "contexts": batch_contexts_agg, "xt": batch_xt_agg, "yt": batch_yt_agg}
