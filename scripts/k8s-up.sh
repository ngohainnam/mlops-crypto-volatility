#!/usr/bin/env bash
# Build the images, load them into a local kind cluster and install both Helm charts.
# Safe to re-run: every step skips or upgrades what already exists.
# Passwords come from .env and go straight into a Kubernetes Secret; nothing secret is in Git.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

CLUSTER=mlops NS=mlops REPO=ghcr.io/ngohainnam/mlops-crypto-volatility

kind get clusters | grep -qx "$CLUSTER" || kind create cluster --config k8s/kind-config.yaml

# kind nodes can't see the host's Docker images, so copy them in (no registry needed).
docker build -f docker/mlflow.Dockerfile -t mlops-mlflow:dev .
docker build -f docker/api.Dockerfile -t "$REPO-api:dev" .
docker build -f docker/trainer.Dockerfile -t "$REPO-trainer:dev" .
kind load docker-image mlops-mlflow:dev "$REPO-api:dev" "$REPO-trainer:dev" --name "$CLUSTER"

helm upgrade --install mlflow helm/mlflow -n "$NS" --create-namespace --wait --timeout 10m \
  --set credentials.postgresUser="$POSTGRES_USER" \
  --set credentials.postgresPassword="$POSTGRES_PASSWORD" \
  --set credentials.postgresDb="$POSTGRES_DB" \
  --set credentials.minioUser="$MINIO_ROOT_USER" \
  --set credentials.minioPassword="$MINIO_ROOT_PASSWORD"
helm upgrade --install api helm/api -n "$NS" --wait --timeout 5m \
  --set binanceBaseUrl="${BINANCE_BASE_URL:-https://api.binance.com}"

kubectl -n "$NS" get pods
