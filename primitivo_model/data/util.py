import lab.torch as B


def cast_task_tensors(task, dtype, device):
    """Cast all tensors in a task to the specified dtype and device.

    Args:
        task: Dictionary containing 'contexts', 'xt', and 'yt' lists of tensors
        dtype: Data type to cast to
        device: Device to place tensors on

    Returns:
        Dictionary with the same structure but with cast tensors
    """
    # Create new lists for each component
    contexts_cast = []
    xt_cast = []
    yt_cast = []

    # Cast context tensors (tuples of x, y tensors)
    for x_ctx, y_ctx in task["contexts"]:
        with B.on_device(device):
            x_ctx_cast = B.cast(dtype, x_ctx)
            y_ctx_cast = B.cast(dtype, y_ctx)
        contexts_cast.append((x_ctx_cast, y_ctx_cast))

    # Cast target input tensors
    for x_trg in task["xt"]:
        with B.on_device(device):
            xt_cast.append(B.cast(dtype, x_trg))

    # Cast target output tensors
    if task["yt"]:
        for y_trg in task["yt"]:
            with B.on_device(device):
                yt_cast.append(B.cast(dtype, y_trg))

        return {"contexts": contexts_cast, "xt": xt_cast, "yt": yt_cast}
    else:
        return {"contexts": contexts_cast, "xt": xt_cast, "yt": None}


def get_xt_extremal_values(xt, max_or_min):
    return [getattr(t, max_or_min)().item() for t in xt if t.shape[-1] > 0]


def get_context_extremal_values(contexts, max_or_min):
    return [getattr(t[0], max_or_min)().item() for t in contexts if t[0].shape[-1] > 0]


def get_task_extremal_values(task, max_or_min: str = "max"):
    extremal_contexts = get_context_extremal_values(task["contexts"], max_or_min)
    extremal_xts = get_xt_extremal_values(task["xt"], max_or_min)

    if max_or_min == "max":
        return max(*extremal_contexts, *extremal_xts)
    elif max_or_min == "min":
        return min(*extremal_contexts, *extremal_xts)
    else:
        raise ValueError(f"Invalid value for max_or_min: {max_or_min}. Use 'max' or 'min'.")


def check_context_empty(context):
    return all([t[0].shape[-1] == 0 for t in context])
