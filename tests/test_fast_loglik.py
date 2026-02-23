"""Tests for the fast_loglik function in primitivo_model.nps.model."""

import pytest
import torch
import neuralprocesses.torch as nps
from neuralprocesses import Aggregate, AggregateInput
import lab.torch as B

from primitivo_model.nps.model import create_model, fast_loglik

# Ignore UserWarnings in these tests
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def model_and_batch():
    """Create model and batch data for testing."""
    model = create_model(num_measurements=2, length_scale=1.0, channels=16, num_layers=2, margin=0.1)
    batch_size, num_context, num_target, num_outputs = 4, 10, 5, 2
    
    # Context for each output dimension - shape: (batch, num_context, 1) for both xc and yc
    contexts = [
        (torch.randn(batch_size, 1, num_context), torch.randn(batch_size, 1, num_context))
        for _ in range(num_outputs)
    ]
    
    # Targets as AggregateInput and Aggregate - shape: (batch, 1, num_target)
    xt = AggregateInput(
        (torch.randn(batch_size, 1, num_target), 0),
        (torch.randn(batch_size, 1, num_target), 1)
    )
    yt = Aggregate(
        torch.randn(batch_size, 1, num_target),
        torch.randn(batch_size, 1, num_target)
    )
    
    batch = {"contexts": contexts, "xt": xt, "yt": yt}
    state = B.create_random_state(torch.float32, seed=42)
    
    return model, batch, state


def compare_with_nps(model, batch, state, **kwargs):
    """Helper to compare fast_loglik with nps.loglik."""
    _, fast_ll = fast_loglik(state, model, batch["contexts"], batch["xt"], batch["yt"], **kwargs)
    _, nps_ll = nps.loglik(state, model, batch["contexts"], batch["xt"], batch["yt"], **kwargs)
    assert torch.allclose(fast_ll, nps_ll, rtol=1e-4, atol=1e-6)
    return fast_ll



def test_with_normalisation(model_and_batch):
    """Test normalisation works correctly."""
    model, batch, state = model_and_batch
    _, ll_unnorm = fast_loglik(state, model, batch["contexts"], batch["xt"], batch["yt"], normalise=False)
    ll_norm = compare_with_nps(model, batch, state, normalise=True)
    assert not torch.allclose(ll_unnorm, ll_norm)


@pytest.mark.parametrize("nan_pattern", [
    "some_nans",      # Sparse NaNs
    "all_nans",       # All values are NaN
    "mixed_nans",     # Different patterns per output
])
def test_nan_handling(model_and_batch, nan_pattern):
    """Test various NaN patterns in targets."""
    model, batch, state = model_and_batch
    yt = [y.clone() for y in batch["yt"]]
    
    if nan_pattern == "some_nans":
        yt[0][0, 0, 0] = yt[1][1, 0, 2] = float("nan")
    elif nan_pattern == "all_nans":
        yt = [torch.full_like(y, float("nan")) for y in yt]
    elif nan_pattern == "mixed_nans":
        yt[0][:, :, :2] = yt[1][:, :, 2:] = float("nan")
    
    batch["yt"] = Aggregate(*yt)
    _, loglik = fast_loglik(state, model, batch["contexts"], batch["xt"], batch["yt"])
    
    assert torch.isfinite(loglik).all()
    if nan_pattern == "all_nans":
        assert torch.allclose(loglik, torch.zeros_like(loglik))
    else:
        compare_with_nps(model, batch, state)


@pytest.mark.parametrize("num_outputs", [1, 10])
def test_different_output_dimensions(num_outputs):
    """Test with varying number of outputs."""
    model = create_model(num_measurements=num_outputs, length_scale=1.0, channels=16, num_layers=2)
    batch_size, num_context, num_target = 4, 10, 5
    
    contexts = [
        (torch.randn(batch_size, 1, num_context), torch.randn(batch_size, 1, num_context))
        for _ in range(num_outputs)
    ]
    
    xt = AggregateInput(*[
        (torch.randn(batch_size, 1, num_target), i) for i in range(num_outputs)
    ])
    yt = Aggregate(*[
        torch.randn(batch_size, 1, num_target) for _ in range(num_outputs)
    ])
    
    batch = {"contexts": contexts, "xt": xt, "yt": yt}
    state = B.create_random_state(torch.float32, seed=42)
    
    loglik = compare_with_nps(model, batch, state)
    assert loglik.shape == (batch_size,) and torch.isfinite(loglik).all()


def test_empty_targets(model_and_batch):
    """Test with no target points."""
    model, _, state = model_and_batch
    batch_size, num_context, num_outputs = 2, 10, 2
    
    contexts = [
        (torch.randn(batch_size, 1, num_context), torch.randn(batch_size, 1, num_context))
        for _ in range(num_outputs)
    ]
    
    xt = AggregateInput(
        (torch.randn(batch_size, 1, 0), 0),
        (torch.randn(batch_size, 1, 0), 1)
    )
    yt = Aggregate(
        torch.randn(batch_size, 1, 0),
        torch.randn(batch_size, 1, 0)
    )
    
    _, loglik = fast_loglik(state, model, contexts, xt, yt)
    assert loglik.shape == (batch_size,) and torch.allclose(loglik, torch.zeros_like(loglik))




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
