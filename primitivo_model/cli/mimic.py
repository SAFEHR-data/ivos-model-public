"""
Command line interface for MIMIC dataset processing.
"""

import typer
from loguru import logger
from rich.console import Console

from primitivo_model.data.mimic.datasets import (
    create_bolton_ivos_labels,
    create_simple_charts_dataset,
    get_abx_rx,
    get_iv_abx_adm,
)
from primitivo_model.data.mimic.preprocess import preprocess_mimic_data
from primitivo_model.db import BaseDb
from primitivo_model.util import print_banner

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


@app.command(name="process", short_help="Process the MIMIC database")
def process(
    smoke_test: bool = typer.Option(
        False,
        "--smoke-test",
        help="Run with a small subset of the data",
    ),
    pre_process: bool = typer.Option(
        False,
        "--pre-process",
        help="Run pre-processing step",
    ),
) -> None:
    """
    Process the MIMIC database and create a new processed version.
    """
    print_banner("## MIMIC Database Processing")

    processed_db = BaseDb(
        name="mimic4", level="processed" + ("-dev" if smoke_test else ""), read_only=not pre_process
    )
    if pre_process:
        mimic_db = BaseDb(name="mimic4", level="", read_only=True)
        # Process the data
        preprocess_mimic_data(db=mimic_db, output_db=processed_db, smoke_test=smoke_test)
        logger.info("MIMIC database pre-processing completed successfully")
        console.print(f"Processed database saved to: {processed_db.path}", style="bold green")

    dataset_db = BaseDb(
        name="mimic4", level="simple-charts" + ("-dev" if smoke_test else ""), read_only=False
    )
    logger.info("Creating simple charts database...")
    create_simple_charts_dataset(processed_db, dataset_db)

    logger.info("Creating on IV cohort...")
    get_iv_abx_adm(processed_db, dataset_db)

    logger.info("Creating on IV labels...")
    get_abx_rx(processed_db, dataset_db)

    logger.info("Creating on Bolton labels...")
    create_bolton_ivos_labels(processed_db, dataset_db)

    logger.info("MIMIC database dataset creation completed successfully")
    console.print(f"Processed database saved to: {dataset_db.path}", style="bold green")
