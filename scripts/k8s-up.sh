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

# Monitoring before ArgoCD: the API chart has a ServiceMonitor/PrometheusRule, whose CRDs come from here.
# The Discord webhook lives only in .env and this Secret, never in Git.
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
kubectl -n monitoring create secret generic alertmanager-discord \
  --from-literal=webhook-url="$DISCORD_WEBHOOK_URL" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n monitoring create configmap volatility-dashboard \
  --from-file=monitoring/grafana/dashboards/volatility.json --dry-run=client -o yaml \
  | kubectl label --local -f - grafana_dashboard=1 -o yaml | kubectl apply -f -
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
helm upgrade --install kps prometheus-community/kube-prometheus-stack --version 92.20.0 \
  -n monitoring --wait --timeout 10m -f k8s/monitoring-values.yaml \
  --set grafana.adminPassword="$GF_SECURITY_ADMIN_PASSWORD"

# The API is deployed by ArgoCD from Git (image pulled from GHCR), never by helm here.
kubectl get ns argocd >/dev/null 2>&1 || kubectl create namespace argocd
kubectl apply -n argocd --server-side --force-conflicts -f https://raw.githubusercontent.com/argoproj/argo-cd/v3.6.1/manifests/install.yaml
kubectl -n argocd rollout status deploy/argocd-repo-server --timeout=5m
kubectl apply -f k8s/argocd-app.yaml
kubectl apply -f k8s/retrain-cronjob.yaml

kubectl -n "$NS" get pods
