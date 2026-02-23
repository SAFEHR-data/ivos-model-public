import io
import os

import matplotlib.pyplot as plt
import mlflow
import torch
from matplotlib.figure import Figure


def configure_mlflow(
    tracking_uri=None,
    registry_uri=None,
):
    """Configure MLFlow with global settings.

    Args:
        tracking_uri: URI for MLFlow tracking server
        registry_uri: URI for model registry
    """
    if tracking_uri is None:
        # Set tracking URI relative to this file's location (2 directories up)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tracking_uri = os.path.join((os.path.dirname(current_dir)), "mlruns")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.enable_system_metrics_logging()
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)
    return mlflow


# Configure MLFlow on module import
configure_mlflow()


def log_plot_to_mlflow(fig, fig_path):
    """
    Updated version to handle matplotlib figures
    """
    if isinstance(fig, Figure):  # If matplotlib figure
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)

        mlflow.log_figure(fig, fig_path)
        plt.close(fig)
    else:  # Original Plotly implementation
        fig.write_image(fig_path)

        mlflow.log_artifact(fig_path)


def log_torch_model_to_mlflow(model, path, run_id):
    local_path = os.path.join(run_id, path)
    os.makedirs(run_id, exist_ok=True)
    torch.save({"weights": model.state_dict()}, local_path)
    mlflow.log_artifact(local_path=local_path)
    os.remove(local_path)  # Clean up local file
    os.rmdir(run_id)  # Remove the run_id directory


def log_sklearn_model_to_mlflow(model, path, run_id):
    local_path = os.path.join(run_id, path)
    os.makedirs(run_id, exist_ok=True)
    model.save_model(local_path)
    mlflow.log_artifact(local_path=local_path)
    os.remove(local_path)  # Clean up local file
    os.rmdir(run_id)  # Remove the run_id directory


def log_text_to_mlflow(text_content, filename, run_id):
    """Log text content as an artifact to MLflow."""
    local_path = os.path.join(run_id, filename)
    os.makedirs(run_id, exist_ok=True)

    with open(local_path, "w") as f:
        f.write(text_content)

    mlflow.log_artifact(local_path=local_path)
    os.remove(local_path)  # Clean up local file
    os.rmdir(run_id)  # Remove the run_id directory


def log_dataframe_to_mlflow(df, filename, run_id):
    """Log DataFrame as CSV artifact to MLflow."""
    local_path = os.path.join(run_id, filename)
    os.makedirs(run_id, exist_ok=True)

    df.to_csv(local_path)

    mlflow.log_artifact(local_path=local_path)
    os.remove(local_path)  # Clean up local file
    os.rmdir(run_id)  # Remove the run_id directory
