# Shortcuts. Run from the project root, for example:  make test
# The lines under each target MUST start with a Tab, not spaces.

.PHONY: data parquet parquet-force splits features test lint format mlflow demo-data \
        demo-data-synthetic api app docker

DEMO = data/processed/demo_applicants.parquet

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
	.venv/bin/ruff check src tests app

format:
	.venv/bin/ruff format src tests app

mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

# Step 9: the demo app. `make app` builds the demo data first if it is missing.
demo-data:
	.venv/bin/python -m creditrisk.serving.demo

# Step 16: made-up applicants for a public demo (no Kaggle rows; safe to commit)
demo-data-synthetic:
	.venv/bin/python -m creditrisk.serving.demo --synthetic

$(DEMO): data/processed/features.parquet models/lightgbm_final.json  # rebuild when either changes
	.venv/bin/python -m creditrisk.serving.demo

api: $(DEMO)
	.venv/bin/uvicorn creditrisk.serving.api:app --reload --port 8000

app: $(DEMO)
	.venv/bin/streamlit run app/streamlit_app.py --server.headless true

# Step 16: the image with the final model and the made-up demo applicants
docker: demo/demo_synthetic.parquet
	docker build -t creditrisk-demo .

demo/demo_synthetic.parquet:
	.venv/bin/python -m creditrisk.serving.demo --synthetic
