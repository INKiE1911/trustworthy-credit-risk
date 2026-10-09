# The demo app (Streamlit, port 8501) and the API (FastAPI, port 8000) in one image.
# By default it holds the stand-in model and the made-up applicants from demo/: nothing real,
# safe to share. MODEL_DIR=models puts the real model in instead: for this machine only.
#   docker build -t creditrisk-demo .                                   (make docker)
#   docker build --build-arg MODEL_DIR=models -t creditrisk-real .      (make docker-real)
#   docker run -p 8501:8501 creditrisk-demo        or        docker compose up
FROM python:3.12-slim

# LightGBM needs OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home appuser
WORKDIR /app
COPY app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ARG MODEL_DIR=demo/model
COPY pyproject.toml ./
COPY configs configs
COPY src src
COPY app app
COPY demo/demo_synthetic.parquet demo/
COPY ${MODEL_DIR} model
RUN chown -R appuser /app
USER appuser

ENV PYTHONPATH=/app/src \
    CREDITRISK_MODEL_DIR=/app/model \
    CREDITRISK_DEMO_PATH=/app/demo/demo_synthetic.parquet \
    MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8501 8000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", \
     "--server.port=8501", "--server.headless=true"]
