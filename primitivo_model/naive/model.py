import lab.torch as B
from primitivo_model.data.generator import TaskSet


def predict(contexts, xt):
    """Always repeat last observed y in contexts for each target xt."""
    means = []
    for (ctx_x, ctx_y), xt_i in zip(contexts, xt):
        xt_i = xt_i[0] if isinstance(xt_i, tuple) else xt_i

        if ctx_y.size > 0:
            # pick last observed value, shape [...,1]
            last_val = ctx_y[..., -1]
            # _preds = B.tile(B.expand_dims(ctx_y[..., -1:], -1), (1, 1, xt_i.shape[-1]))
            pred = B.ones(*xt_i.shape) * last_val
        else:
            pred = B.zeros(*xt_i.shape)

        means.append(pred)
    # mirror NPs signature: return state, means, placeholders...
    return means, None, None, None


def mae_objective(batch):
    """Compute MAE of repeat‐last preds over a batch."""
    means, *_ = predict(batch["contexts"], batch["xt"])
    errs = [(B.abs(m - y)) for m, y in zip(means, batch["yt"])]
    val = B.mean(B.concat(*errs, axis=2))
    return val


def eval_epoch(task_set: TaskSet):
    losses = []
    for task_id, batch in task_set.tasks.items():
        loss = mae_objective(batch)
        losses.append(B.reshape(loss, 1))
    return B.mean(B.concat(*losses))


def collect_predictions(task_set: TaskSet):
    """Collect predictions for all tasks in the task set."""
    task_preds = {}

    for task_id, task in task_set.tasks.items():
        # Get predictions for this task
        means, *_ = predict(task["contexts"], task["xt"])
        task_preds[task_id] = means

    return task_preds
