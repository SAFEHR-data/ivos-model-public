import neuralprocesses.torch as nps
import numpy as np
import torch
from loguru import logger
from neuralprocesses import MultiOutputNormal, num_data
from tqdm import tqdm

import lab.torch as B
from primitivo_model.mlflow_utils import log_torch_model_to_mlflow, mlflow

# from primitivo_model.plots import make_and_log_task_plots, plot_predict_fn_task


def create_lr_scheduler(optimizer, num_epochs, learning_rate):
    """
    Create a learning rate scheduler with linear warmup and cosine annealing.

    Warmup: Linear increase from 1% to 100% of LR over 10% of total epochs
    Main: Cosine annealing from 100% to 0.1% of LR over remaining 90% of epochs

    Args:
        optimizer: The optimizer to schedule
        num_epochs: Total number of training epochs
        learning_rate: Base learning rate

    Returns:
        A SequentialLR scheduler combining warmup and cosine annealing
    """
    warmup_epochs = int(0.1 * num_epochs)  # 10% of total epochs for warmup

    # Linear warmup scheduler: from 1% to 100% of LR
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,  # Start at 1% of base LR
        end_factor=1.0,  # End at 100% of base LR
        total_iters=warmup_epochs,
    )

    # Cosine annealing scheduler: from 100% to 0.1% of LR
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,  # Remaining epochs after warmup
        eta_min=learning_rate * 0.001,  # End at 0.1% of base LR
    )

    # Combine schedulers using SequentialLR
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],  # Switch from warmup to cosine at this epoch
    )

    return scheduler


def create_model(
    num_measurements, length_scale, channels, num_layers, margin=None, dtype=torch.float32
):
    return nps.construct_convgnp(
        dim_x=1,
        dim_y=num_measurements,
        dim_yc=(1,) * num_measurements,
        likelihood="het",  # ConvCNP model
        points_per_unit=2 / length_scale,
        unet_channels=(channels,) * num_layers,
        margin=0.1 if margin is None else margin,
        dtype=dtype,
    )


def fast_loglik(
    state,
    model,
    contexts,
    xt,
    yt,
    normalise=False,
    dtype_lik=None,
):
    # Get predictions from model
    state, pred = model(state, contexts, xt)

    # Get appropriate dtypes for likelihood computation
    float = B.dtype_float(yt)
    float64 = B.promote_dtypes(float, np.float64)

    if not dtype_lik:
        dtype_lik = float64

    # Compute log-likelihood for each output dimension
    batch_shape = B.shape(yt[0], 0)
    summed_logprob = B.zeros(float64, batch_shape)
    # loop over outputs
    for yt_i, mean_i, var_i in zip(yt, pred.mean, pred.var):
        if yt_i.shape[-1] == 0:
            # output contains no targets
            continue

        # Replace NaNs with zeros for computation
        yt_clean = torch.nan_to_num(yt_i)

        # Create mask for valid (non-NaN) observations
        valid_mask = (~torch.isnan(yt_i)).float()

        # Compute log-probability using diagonal multivariate normal by flattening everything
        logprob = MultiOutputNormal.diagonal(
            mean_i.reshape(-1, 1), var_i.reshape(-1, 1), shape=(1,)
        ).logpdf(yt_clean.reshape(-1, 1))

        # Apply mask to set NaN positions to 0.0
        logprob_masked = logprob * valid_mask.reshape(-1)

        # Sum over data points, keeping batch dimension, squeezing output dim (always 1)
        summed_logprob += B.squeeze(B.sum(logprob_masked.reshape(B.shape(yt_i)), axis=-1))

    # Optionally normalize by number of valid data points
    if normalise:
        n_data = B.cast(float64, num_data(xt, yt))
        summed_logprob = summed_logprob / n_data

    return state, summed_logprob


def fast_loglike_objective(state, model, batch, normalise):
    state, obj = fast_loglik(
        state,
        model,
        batch["contexts"],
        batch["xt"],
        batch["yt"],
        normalise=normalise,
    )
    # print(obj)

    val = -B.mean(obj)

    return state, val


def loglike_objective(state, model, batch, normalise):
    state, obj = nps.loglik(
        state,
        model,
        batch["contexts"],
        batch["xt"],
        batch["yt"],
        normalise=normalise,
    )

    val = -B.mean(obj)

    return state, val


def ar_loglike_objective(state, model, batch, normalise):
    state, obj = nps.ar_loglik(
        state,
        model,
        batch["contexts"],
        batch["xt"],
        batch["yt"],
        normalise=normalise,
        order="given",
    )
    # print(obj)
    val = -B.mean(obj)
    return state, val


def mae_objective(state, model, batch):
    state, mean, _, _, _ = nps.predict(state, model, batch["contexts"], batch["xt"])
    val = B.concat(*(B.abs(pred - true) for pred, true in zip(mean, batch["yt"])), axis=2)
    # data_per_batch = nps.num_data(batch["xt"], batch["yt"])
    # num_nans = B.sum(~B.isnan(val), axis=2)
    # print(data_per_batch)
    # print(num_nans)
    # mean over both measurement and batch dimensions
    val = B.mean(B.nanmean(val, axis=2))
    return state, val


def train_epoch(state, model, train_gen, objective, opt):
    """Perform a training epoch."""
    model.train()
    for batch in train_gen.epoch():
        state, val = objective(state, model, batch)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()

    return val, state


def eval_epoch(state, model, val_gen, objective):
    """Perform a eval epoch."""
    model.eval()
    total_examples = 0
    with torch.no_grad():
        vals = []
        for batch in val_gen.epoch():
            batch_size = len(batch["ids"])

            state, val = objective(state, model, batch)
            val = val * batch_size
            vals.append(B.reshape(val, 1))

            total_examples += batch_size

        # if take_mean:
        obj_mean = B.sum(B.concat(*vals, axis=0)) / total_examples
        return obj_mean, state


def fit_model(
    state,
    model,
    train_batches,
    eval_batches,
    objective,
    learning_rate=1e-3,
    num_epochs=10,
    patience=5,  # New parameter for early stopping
    epsilon=1e-8,
    use_lr_scheduler=False,  # New parameter to enable/disable scheduler
):
    # Initialize optimizer
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    run_id = mlflow.active_run().info.run_id

    # Initialize learning rate scheduler if requested
    scheduler = None
    if use_lr_scheduler:
        scheduler = create_lr_scheduler(opt, num_epochs, learning_rate)
        logger.info("Using learning rate scheduler with warmup and cosine annealing")

    B.epsilon = 1e-2
    best_eval_lik = np.inf
    patience_counter = 0  # Counter for early stopping

    # Run the training loop.
    progress_bar = tqdm(range(num_epochs), desc="Epochs")
    try:
        for epoch in progress_bar:
            if epoch > 0:
                B.epsilon = epsilon
            # Compute training objective.
            train_obj, state = train_epoch(state, model, train_batches, objective, opt)
            # Update epoch progress bar with current loss

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_obj.item(), step=epoch)
            log_torch_model_to_mlflow(model, "model-last.torch", run_id)

            # Log current learning rate
            current_lr = opt.param_groups[0]["lr"]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            eval_obj, state = eval_epoch(state, model, eval_batches, objective)
            mlflow.log_metric("eval_loss", eval_obj.item(), step=epoch)

            if eval_obj < best_eval_lik:
                best_eval_lik = eval_obj
                mlflow.log_metric("best_eval_loss", best_eval_lik.item())
                patience_counter = 0  # Reset patience counter
                # Save the model with the best evaluation loss
                log_torch_model_to_mlflow(model, "model-best.torch", run_id)
            else:
                patience_counter += 1  # Increment patience counter

            # Check for early stopping
            if patience_counter >= patience:
                raise KeyboardInterrupt

            # Step the learning rate scheduler if enabled
            if scheduler is not None:
                scheduler.step()

            progress_bar.set_postfix(
                {
                    "train_loss": f"{train_obj.item():.4f}",
                    "eval_loss": f"{eval_obj.item():.4f}",
                    "best_eval": f"{best_eval_lik:.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

    except KeyboardInterrupt:
        logger.info("Training stopping early...")
        # Log that training was interrupted
        mlflow.log_param("training_interrupted", True)
        mlflow.log_param("completed_epochs", epoch + 1)

    else:
        mlflow.log_param("training_interrupted", False)
        mlflow.log_param("completed_epochs", num_epochs)

    return model, state
