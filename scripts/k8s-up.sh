#!/usr/bin/env bash
# Create the kind cluster, install the MLflow chart, and hand the API over to ArgoCD.
# Safe to re-run: every step skips or upgrades what already exists.
# Passwords come from .env and go straight into a Kubernetes Secret; nothing secret is in Git.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

CLUSTER=mlops NS=mlops REPO=ghcr.io/ngohainnam/mlops-crypto-volatility

kind get clusters | grep -qx "$CLUSTER" || kind create cluster --config k8s/kind-config.yaml

# kind nodes can't see the host's Docker images, so copy them in (no registry needed).
docker build -f docker/mlflow.Dockerfile -t mlops-mlflow:dev .
docker build -f docker/trainer.Dockerfile -t "$REPO-trainer:dev" .
kind load docker-image mlops-mlflow:dev "$REPO-trainer:dev" --name "$CLUSTER"

helm upgrade --install mlflow helm/mlflow -n "$NS" --create-namespace --wait --timeout 10m \
  --set credentials.postgresUser="$POSTGRES_USER" \
  --set credentials.postgresPassword="$POSTGRES_PASSWORD" \
  --set credentials.postgresDb="$POSTGRES_DB" \
  --set credentials.minioUser="$MINIO_ROOT_USER" \
  --set credentials.minioPassword="$MINIO_ROOT_PASSWORD"
# The API is deployed by ArgoCD from Git (image pulled from GHCR), never by helm here.
kubectl get ns argocd >/dev/null 2>&1 || kubectl create namespace argocd
kubectl apply -n argocd --server-side --force-conflicts -f https://raw.githubusercontent.com/argoproj/argo-cd/v3.6.1/manifests/install.yaml
kubectl -n argocd rollout status deploy/argocd-repo-server --timeout=5m
kubectl apply -f k8s/argocd-app.yaml

kubectl -n "$NS" get pods
