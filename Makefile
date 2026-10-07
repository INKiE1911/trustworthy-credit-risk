# Shortcuts. Run from the project root, for example:  make test
# The lines under each target MUST start with a Tab, not spaces.

.PHONY: data parquet parquet-force splits features test lint format mlflow

data:
	mkdir -p data/raw
	.venv/bin/kaggle competitions download -c home-credit-default-risk -p data/raw
	unzip -o -q data/raw/home-credit-default-risk.zip -d data/raw

parquet:
	.venv/bin/python -m creditrisk.data.to_parquet

parquet-force:
	.venv/bin/python -m creditrisk.data.to_parquet --force

splits:
	.venv/bin/python -m creditrisk.data.split

features:
	.venv/bin/python -m creditrisk.features.build

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests

format:
	.venv/bin/ruff format src tests

mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
