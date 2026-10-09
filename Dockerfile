# The demo app (Streamlit, port 8501) and the API (FastAPI, port 8000) in one image.
# It holds the final model and the made-up demo applicants only: no Kaggle rows (see .dockerignore).
#   docker build -t creditrisk-demo .
#   docker run -p 8501:8501 creditrisk-demo          # the app
#   docker compose up                                # app + API together
FROM python:3.12-slim

# LightGBM needs OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home appuser
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY pyproject.toml ./
COPY configs configs
COPY src src
COPY app app
COPY models models
COPY demo demo
RUN chown -R appuser /app
USER appuser

ENV PYTHONPATH=/app/src \
    CREDITRISK_DEMO_PATH=/app/demo/demo_synthetic.parquet \
    MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8501 8000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", \
     "--server.port=8501", "--server.headless=true"]
