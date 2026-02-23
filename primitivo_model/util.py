import numpy as np
from rich.console import Console
from rich.markdown import Markdown


def print_banner(text) -> None:
    console = Console()
    console.print(Markdown(f"{text}"), style="bold yellow")


def convert_str_to_float_or_none(x):
    return None if x.lower() == "none" else float(x)


def to_numpy(x, squeeze=True):
    """Convert a PyTorch tensor to NumPy."""

    if squeeze:
        x = x.squeeze()

    if isinstance(x, np.ndarray) or isinstance(x, float):
        return x

    return x.detach().cpu().numpy()
