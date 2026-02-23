# Primitivo Model - Probabilistic Prescribing

Probabilistic prescribing model for IV-to-Oral antibiotic switching predictions. Uses Neural Processes (ConvGNP), GBDT/Logistic Regression, and baseline models trained on MIMIC-IV healthcare time series data.

## Setup

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for Python package management.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install

```shell
git clone https://github.com/SAFEHR-data/ivos-model.git
cd ivos-model
uv sync --all-extras
uv run pre-commit install
```

## Usage

```shell
# Show CLI help
uv run primitivo-model --help

# Train models
uv run primitivo-model nps train --route simple-charts-dev --data-source mimic4
uv run primitivo-model tabular train-gbdt --route simple-charts-dev --data-source mimic4
uv run primitivo-model baseline repeat-last --route simple-charts-dev --data-source mimic4

# Process MIMIC data
uv run primitivo-model mimic process --smoke-test

# Show configuration
uv run primitivo-model config
```

### Running Tests

```shell
uv run pytest
uv run pytest tests/test_criteria.py -v
```

### Linting and Formatting

```shell
uv run ruff check .
uv run ruff format .
uv run pre-commit run --all-files
```

## MLflow Integration

All experiments log to a local `mlruns/` directory.

```shell
mlflow ui --port 5000
```

## Data

### MIMIC-IV

The models are trained on [MIMIC-IV](https://physionet.org/content/mimiciv/), which requires credentialed access via PhysioNet. To prepare the database:

1. **Download the CSV files** (requires a PhysioNet account with signed data use agreement):
   ```shell
   wget -r -N -c -np --user YOUR_USERNAME --ask-password \
       https://physionet.org/files/mimiciv/3.1/
   ```

2. **Build the DuckDB database** using [mimic-code](https://github.com/MIT-LCP/mimic-code):
   ```shell
   git clone https://github.com/MIT-LCP/mimic-code
   # Follow the instructions at:
   # https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/duckdb
   ```
   > **Note for v3.1:** CSV parsing requires a small fix — on line 115 of the build script, change:
   > ```sql
   > COPY $TABLE_NAME FROM '$FILE' (HEADER);
   > ```
   > to:
   > ```sql
   > COPY $TABLE_NAME FROM '$FILE' (DELIMITER ',', HEADER, ESCAPE '"');
   > ```
   > See [mimic-code#1881](https://github.com/MIT-LCP/mimic-code/issues/1881) for details.

3. **Validate the database:**
   ```shell
   duckdb mimic4.db < mimic-code/mimic-iv/postgres/validate.sql
   ```

4. **Process into model format**, pointing `DATA_ROOT` at the directory containing `mimic4/mimic4.db`:
   ```shell
   export DATA_ROOT=/path/to/your/data
   uv run primitivo-model mimic process
   ```

By default `DATA_ROOT` is the `data/` directory in the project root.

### Radix

The Radix dataset used in this project is not publicly available.

