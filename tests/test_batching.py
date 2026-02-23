import pytest
import torch
import lab as B
from unittest.mock import Mock, MagicMock
from primitivo_model.nps.data import TaskLoader, create_batch_from_tasks
from primitivo_model.data.generator import TaskSet
from neuralprocesses import AggregateInput, Aggregate


class MockTaskSet(TaskSet):
    """Mock TaskSet for testing purposes."""
    
    def __init__(self, num_tasks=100, num_measurements=3):
        self.num_tasks = num_tasks
        self.num_measurements = num_measurements
        self.measurement_names = [f"measurement_{i}" for i in range(num_measurements)]
        
        # Create mock encounter IDs
        self.encounter_ids = [f"task_{i:03d}" for i in range(num_tasks)]
        
        # Create mock tasks
        self.tasks = {}
        for i, enc_id in enumerate(self.encounter_ids):
            self.tasks[enc_id] = {
                "contexts": [(torch.randn(1, 1, 10), torch.randn(1, 1, 10)) 
                           for _ in range(num_measurements)],
                "xt": [torch.randn(1, 1, 5) for _ in range(num_measurements)],
                "yt": [torch.randn(1, 1, 5) for _ in range(num_measurements)],
            }
    
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, key):
        return self.tasks[key]


class TestTaskLoader:
    """Test suite for TaskLoader functionality."""
    
    @pytest.fixture
    def mock_task_set(self):
        """Create a mock task set with 100 tasks."""
        return MockTaskSet(num_tasks=100, num_measurements=3)
    
    @pytest.fixture
    def device(self):
        """Use CPU for testing."""
        return torch.device("cpu")
    
    def test_basic_initialization(self, mock_task_set, device):
        """Test basic TaskLoader initialization."""
        loader = TaskLoader(
            task_set=mock_task_set,
            batch_size=10,
            device=device,
            seed=42
        )
        
        assert loader.batch_size == 10
        assert loader.device == device
        assert loader.num_unique_batches == 10  # 100 tasks / 10 batch_size
        assert loader.num_batches == 10  # No epoch_size specified
        assert len(loader.batches) == 10
    
    def test_epoch_size_larger_than_dataset(self, mock_task_set, device):
        """Test when epoch_size is larger than dataset size."""
        loader = TaskLoader(
            task_set=mock_task_set,
            batch_size=10,
            device=device,
            epoch_size=200,  # Larger than 100 tasks
            seed=42
        )
        
        assert loader.num_unique_batches == 10  # Still 10 unique batches
        assert loader.num_batches == 20  # 200 / 10 batch_size
        
        # Test that we can iterate through full epoch
        all_batches = list(loader.epoch())
        assert len(all_batches) == 20
        
        # Collect all task IDs from epoch
        all_task_ids = []
        for batch in all_batches:
            all_task_ids.extend(batch["ids"])
        
        # Should have resampling (more IDs than unique tasks)
        assert len(all_task_ids) == 200
        unique_task_ids = set(all_task_ids)
        assert len(unique_task_ids) <= 100  # Can't have more unique than exist
    
    def test_epoch_size_smaller_than_dataset(self, mock_task_set, device):
        """Test when epoch_size is smaller than dataset size."""
        loader = TaskLoader(
            task_set=mock_task_set,
            batch_size=10,
            device=device,
            epoch_size=50,  # Smaller than 100 tasks
            seed=42
        )
        
        assert loader.num_unique_batches == 10  # Still 10 unique batches available
        assert loader.num_batches == 5  # 50 / 10 batch_size
        
        # Test that we can iterate through full epoch
        all_batches = list(loader.epoch())
        assert len(all_batches) == 5
        
        # Collect all task IDs from epoch
        all_task_ids = []
        for batch in all_batches:
            all_task_ids.extend(batch["ids"])
        
        # Should have exactly 50 tasks, all unique
        assert len(all_task_ids) == 50
        assert len(set(all_task_ids)) == 50
    
    def test_no_epoch_size_uses_full_dataset(self, mock_task_set, device):
        """Test that when epoch_size=None, we use the full dataset exactly once."""
        loader = TaskLoader(
            task_set=mock_task_set,
            batch_size=10,
            device=device,
            seed=42
        )
        
        # Test the critical condition we were debugging
        all_batches = list(loader.epoch())
        all_task_ids = [task_id for batch in all_batches for task_id in batch["ids"]]
        
        # This is the key test - should see each task exactly once
        assert len(all_task_ids) == 100
        assert len(set(all_task_ids)) == 100
        assert set(all_task_ids) == set(mock_task_set.encounter_ids)
    
    def test_incomplete_last_batch(self, device):
        """Test handling of incomplete last batch."""
        # Create task set with non-divisible number of tasks
        task_set = MockTaskSet(num_tasks=95, num_measurements=3)
        loader = TaskLoader(
            task_set=task_set,
            batch_size=10,
            device=device,
            seed=42
        )
        
        assert loader.num_unique_batches == 10  # 9 full batches + 1 partial (5 tasks)
        
        all_batches = list(loader.epoch())
        assert len(all_batches) == 10
        
        # Check batch sizes
        batch_sizes = [len(batch["ids"]) for batch in all_batches]
        assert batch_sizes.count(10) == 9  # 9 full batches
        assert batch_sizes.count(5) == 1   # 1 partial batch
        
        # Check we see all tasks exactly once
        all_task_ids = [task_id for batch in all_batches for task_id in batch["ids"]]
        assert len(all_task_ids) == 95
        assert len(set(all_task_ids)) == 95
    
    def test_dataset_fraction(self, mock_task_set, device):
        """Test dataset_fraction parameter."""
        loader = TaskLoader(
            task_set=mock_task_set,
            batch_size=10,
            device=device,
            dataset_fraction=0.5,  # Use only 50% of data
            seed=42
        )
        
        # Should only use 50 tasks
        all_batches = list(loader.epoch())
        all_task_ids = [task_id for batch in all_batches for task_id in batch["ids"]]
        
        assert len(all_task_ids) == 50
        assert len(set(all_task_ids)) == 50
        
        # All task IDs should be from the original set
        assert set(all_task_ids).issubset(set(mock_task_set.encounter_ids))
    
    def test_multiple_epochs_different_order(self, mock_task_set, device):
        """Test that multiple epochs produce different orderings."""
        loader = TaskLoader(
            task_set=mock_task_set,
            batch_size=10,
            device=device,
            seed=42
        )
        
        # Get task IDs from two consecutive epochs
        epoch1_ids = [task_id for batch in loader.epoch() for task_id in batch["ids"]]
        epoch2_ids = [task_id for batch in loader.epoch() for task_id in batch["ids"]]
        
        # Both should have same tasks but likely different order
        assert set(epoch1_ids) == set(epoch2_ids)
        assert len(epoch1_ids) == len(epoch2_ids) == 100
        
        # Order should be different (with high probability)
        assert epoch1_ids != epoch2_ids
    
    def test_reproducibility_with_same_seed(self, mock_task_set, device):
        """Test that same seed produces same results."""
        loader1 = TaskLoader(
            task_set=mock_task_set,
            batch_size=10,
            device=device,
            seed=42
        )
        
        loader2 = TaskLoader(
            task_set=mock_task_set,
            batch_size=10,
            device=device,
            seed=42
        )
        
        epoch1_ids = [task_id for batch in loader1.epoch() for task_id in batch["ids"]]
        epoch2_ids = [task_id for batch in loader2.epoch() for task_id in batch["ids"]]
        
        assert epoch1_ids == epoch2_ids
    
    def test_edge_case_single_batch(self, device):
        """Test edge case where entire dataset fits in one batch."""
        task_set = MockTaskSet(num_tasks=5, num_measurements=3)
        loader = TaskLoader(
            task_set=task_set,
            batch_size=10,  # Larger than dataset
            device=device,
            seed=42
        )
        
        assert loader.num_unique_batches == 1
        
        all_batches = list(loader.epoch())
        assert len(all_batches) == 1
        assert len(all_batches[0]["ids"]) == 5
    
    def test_edge_case_empty_dataset(self, device):
        """Test edge case with empty dataset."""
        task_set = MockTaskSet(num_tasks=0, num_measurements=3)
        loader = TaskLoader(
            task_set=task_set,
            batch_size=10,
            device=device,
            seed=42
        )
        
        assert loader.num_unique_batches == 0
        assert loader.num_batches == 0
        
        all_batches = list(loader.epoch())
        assert len(all_batches) == 0
    
    def test_batch_structure(self, mock_task_set, device):
        """Test that batches have correct structure."""
        loader = TaskLoader(
            task_set=mock_task_set,
            batch_size=5,
            device=device,
            seed=42
        )
        
        batch = next(iter(loader.epoch()))
        
        # Check batch structure
        assert "ids" in batch
        assert "contexts" in batch
        assert "xt" in batch
        assert "yt" in batch
        
        # Check batch content
        assert len(batch["ids"]) == 5
        assert len(batch["contexts"]) == 3  # num_measurements
        
        # Check contexts structure
        for context in batch["contexts"]:
            assert isinstance(context, tuple)
            assert len(context) == 2  # (x, y)


# Move standalone tests into the class as well
class TestTaskLoaderBugScenarios:
    """Test specific bug scenarios that were causing issues."""
    
    
    @pytest.mark.parametrize("num_tasks,batch_size", [
        (100, 10),    # Even division
        (95, 10),     # Incomplete last batch
        (857, 64),    # Original bug scenario
        (50, 100),    # Single batch
        (1000, 37),   # Prime batch size
    ])
    def test_various_task_batch_combinations(self, num_tasks, batch_size):
        """Test various combinations of task counts and batch sizes."""
        task_set = MockTaskSet(num_tasks=num_tasks, num_measurements=3)
        loader = TaskLoader(
            task_set=task_set,
            batch_size=batch_size,
            device=torch.device("cpu"),
            seed=42
        )
        
        batch_task_ids = [task_id for batch in loader.epoch() for task_id in batch["ids"]]
        
        # Key assertions that should always hold
        assert len(batch_task_ids) == num_tasks
        assert len(set(batch_task_ids)) == num_tasks
        assert set(batch_task_ids) == set(task_set.encounter_ids)


class TestCreateBatchFromTasks:
    """Test the create_batch_from_tasks function."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    @pytest.fixture
    def dtype(self):
        return torch.float32
    
    @pytest.fixture
    def measurement_names(self):
        return ["hr", "temp", "resp"]
    
    @pytest.fixture
    def simple_tasks(self):
        """Create simple tasks with uniform shapes."""
        return {
            "task_001": {
                "contexts": [
                    (torch.randn(1, 1, 10), torch.randn(1, 1, 10)),
                    (torch.randn(1, 1, 10), torch.randn(1, 1, 10)),
                    (torch.randn(1, 1, 10), torch.randn(1, 1, 10)),
                ],
                "xt": [torch.randn(1, 1, 5), torch.randn(1, 1, 5), torch.randn(1, 1, 5)],
                "yt": [torch.randn(1, 1, 5), torch.randn(1, 1, 5), torch.randn(1, 1, 5)],
            },
            "task_002": {
                "contexts": [
                    (torch.randn(1, 1, 10), torch.randn(1, 1, 10)),
                    (torch.randn(1, 1, 10), torch.randn(1, 1, 10)),
                    (torch.randn(1, 1, 10), torch.randn(1, 1, 10)),
                ],
                "xt": [torch.randn(1, 1, 5), torch.randn(1, 1, 5), torch.randn(1, 1, 5)],
                "yt": [torch.randn(1, 1, 5), torch.randn(1, 1, 5), torch.randn(1, 1, 5)],
            },
        }
    
    def test_basic_batch_creation(self, simple_tasks, dtype, device, measurement_names):
        """Test basic batch creation with uniform shapes."""
        enc_ids = ["task_001", "task_002"]
        batch = create_batch_from_tasks(simple_tasks, enc_ids, dtype, device, measurement_names, no_yt=False)
        
        assert batch["ids"] == enc_ids
        assert len(batch["contexts"]) == 3
        assert isinstance(batch["xt"], AggregateInput)
        assert isinstance(batch["yt"], Aggregate)
    
    def test_no_yt_flag(self, simple_tasks, dtype, device, measurement_names):
        """Test that no_yt=True returns None for yt."""
        enc_ids = ["task_001"]
        batch = create_batch_from_tasks(simple_tasks, enc_ids, dtype, device, measurement_names, no_yt=True)
        
        assert batch["yt"] is None
        assert batch["xt"] is not None
    
    def test_padding_with_variable_lengths(self, dtype, device, measurement_names):
        """Test padding when tasks have different sequence lengths."""
        tasks = {
            "task_001": {
                "contexts": [
                    (torch.randn(1, 1, 10), torch.randn(1, 1, 10)),
                    (torch.randn(1, 1, 10), torch.randn(1, 1, 10)),
                    (torch.randn(1, 1, 10), torch.randn(1, 1, 10)),
                ],
                "xt": [torch.randn(1, 1, 3), torch.randn(1, 1, 5), torch.randn(1, 1, 2)],
                "yt": [torch.randn(1, 1, 3), torch.randn(1, 1, 5), torch.randn(1, 1, 2)],
            },
            "task_002": {
                "contexts": [
                    (torch.randn(1, 1, 10), torch.randn(1, 1, 10)),
                    (torch.randn(1, 1, 10), torch.randn(1, 1, 10)),
                    (torch.randn(1, 1, 10), torch.randn(1, 1, 10)),
                ],
                "xt": [torch.randn(1, 1, 7), torch.randn(1, 1, 2), torch.randn(1, 1, 8)],
                "yt": [torch.randn(1, 1, 7), torch.randn(1, 1, 2), torch.randn(1, 1, 8)],
            },
        }
        
        enc_ids = ["task_001", "task_002"]
        batch = create_batch_from_tasks(tasks, enc_ids, dtype, device, measurement_names, no_yt=False)
        
        # Check that all xt tensors are padded to max length for each measurement
        for i, (xt_tensor, _) in enumerate(batch["xt"]):
            assert xt_tensor.shape[0] == 2  # Batch size
            assert xt_tensor.shape[1] == 1  # Channel dimension
            # Shape[2] should be max across both tasks for this measurement
            expected_max = max(tasks["task_001"]["xt"][i].shape[2], tasks["task_002"]["xt"][i].shape[2])
            assert xt_tensor.shape[2] == expected_max
        
        # Check yt padding
        for i, yt_tensor in enumerate(batch["yt"]):
            assert yt_tensor.shape[0] == 2
            expected_max = max(tasks["task_001"]["yt"][i].shape[2], tasks["task_002"]["yt"][i].shape[2])
            assert yt_tensor.shape[2] == expected_max
    
    def test_single_task_batch(self, simple_tasks, dtype, device, measurement_names):
        """Test batch creation with a single task."""
        enc_ids = ["task_001"]
        batch = create_batch_from_tasks(simple_tasks, enc_ids, dtype, device, measurement_names, no_yt=False)
        
        assert len(batch["ids"]) == 1
        for xt_tensor, _ in batch["xt"]:
            assert xt_tensor.shape[0] == 1  # Batch size of 1
    
    def test_context_merging(self, simple_tasks, dtype, device, measurement_names):
        """Test that contexts are properly merged."""
        enc_ids = ["task_001", "task_002"]
        batch = create_batch_from_tasks(simple_tasks, enc_ids, dtype, device, measurement_names, no_yt=False)
        
        # Each context should be a tuple of (xc, yc)
        for context in batch["contexts"]:
            assert isinstance(context, tuple)
            assert len(context) == 2
            xc, yc = context
            assert xc.shape[0] == 2  # Batch size
            assert yc.mask.shape[0] == 2


# Pytest runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])