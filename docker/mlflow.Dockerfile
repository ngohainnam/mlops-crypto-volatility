# Official MLflow image, same version as uv.lock, plus the Postgres and S3 drivers it needs.
FROM ghcr.io/mlflow/mlflow:v3.18.0
RUN pip install --no-cache-dir "psycopg[binary]" boto3 \
 && useradd --create-home --uid 1000 mlflow
USER mlflow
