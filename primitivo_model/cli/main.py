import os

import typer
from rich.console import Console
from rich.markdown import Markdown

os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
import warnings

from primitivo_model.cli import ar_sampling, baseline, mimic, nps, tabular
from primitivo_model.config import settings
from primitivo_model.util import print_banner

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=UserWarning)


app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(nps.app, name="nps", short_help="Neural process state forecasting model")
app.add_typer(baseline.app, name="baseline", short_help="Baseline  state forecasting model")
app.add_typer(
    tabular.app, name="tabular", short_help="Tabular models for direct criteria prediction"
)
app.add_typer(ar_sampling.app, name="ar-sampling", short_help="AR sampling evaluation experiments")
app.add_typer(mimic.app, name="mimic", short_help="Process and analyze MIMIC database")


@app.command(name="config", short_help="Show configuration settings")
def config() -> None:
    console = Console()
    console.print(settings.as_md())
    console.print(Markdown("---"), style="bold")


def run() -> None:
    print_banner("# Primitivo Model")
    app()


if __name__ == "__main__":
    run()
