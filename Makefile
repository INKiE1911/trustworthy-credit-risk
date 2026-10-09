# Shortcuts. Run from the project root, for example:  make test
# The lines under each target MUST start with a Tab, not spaces.

.PHONY: data parquet parquet-force splits features test lint format mlflow demo-data \
        demo-data-synthetic api app docker docker-real report

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

# Step 16: the image. Default: stand-in model + made-up applicants (safe to share).
docker: demo/demo_synthetic.parquet
	docker build -t creditrisk-demo .

# The real model inside, for this machine only: never push this image anywhere.
docker-real: models/lightgbm_final.joblib demo/demo_synthetic.parquet
	docker build --build-arg MODEL_DIR=models -t creditrisk-real .

demo/demo_synthetic.parquet:
	.venv/bin/python -m creditrisk.serving.demo --synthetic

# Step 17: the report -> reports/report.pdf (needs pdflatex + bibtex)
report:
	.venv/bin/python reports/report/system_diagram.py
	cd reports/report && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null \
		&& bibtex report >/dev/null && pdflatex -interaction=nonstopmode report.tex >/dev/null \
		&& pdflatex -interaction=nonstopmode -halt-on-error report.tex | grep -E "Warning|Output"
	cp reports/report/report.pdf reports/report.pdf
