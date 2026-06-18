#!/bin/bash
# Creates GCP log-based metrics and monitoring dashboard for Fake News Detection model.
# Run once after deploying the Cloud Run service.
#
# Usage:
#   export GCP_PROJECT_ID=your-project-id
#   export CLOUD_RUN_SERVICE=your-service-name
#   bash scripts/setup_monitoring.sh

set -euo pipefail

: "${GCP_PROJECT_ID:?GCP_PROJECT_ID is required}"
: "${CLOUD_RUN_SERVICE:?CLOUD_RUN_SERVICE is required}"

LOG_FILTER_BASE="resource.type=\"cloud_run_revision\" resource.labels.service_name=\"${CLOUD_RUN_SERVICE}\""

echo "=== Creating log-based metrics ==="

# Metric 1: Fake news predictions
gcloud logging metrics create fake_prediction_count \
  --project="${GCP_PROJECT_ID}" \
  --description="Number of FAKE news predictions from /api/predict" \
  --log-filter="${LOG_FILTER_BASE} jsonPayload.event=\"prediction\" jsonPayload.is_fake=true" \
  2>/dev/null || \
gcloud logging metrics update fake_prediction_count \
  --project="${GCP_PROJECT_ID}" \
  --description="Number of FAKE news predictions from /api/predict" \
  --log-filter="${LOG_FILTER_BASE} jsonPayload.event=\"prediction\" jsonPayload.is_fake=true"

echo "  [OK] fake_prediction_count"

# Metric 2: Real news predictions
gcloud logging metrics create real_prediction_count \
  --project="${GCP_PROJECT_ID}" \
  --description="Number of REAL news predictions from /api/predict" \
  --log-filter="${LOG_FILTER_BASE} jsonPayload.event=\"prediction\" jsonPayload.is_fake=false" \
  2>/dev/null || \
gcloud logging metrics update real_prediction_count \
  --project="${GCP_PROJECT_ID}" \
  --description="Number of REAL news predictions from /api/predict" \
  --log-filter="${LOG_FILTER_BASE} jsonPayload.event=\"prediction\" jsonPayload.is_fake=false"

echo "  [OK] real_prediction_count"

# Metric 3: Prediction errors
gcloud logging metrics create prediction_error_count \
  --project="${GCP_PROJECT_ID}" \
  --description="Number of prediction errors from /api/predict" \
  --log-filter="${LOG_FILTER_BASE} jsonPayload.event=\"prediction_error\"" \
  2>/dev/null || \
gcloud logging metrics update prediction_error_count \
  --project="${GCP_PROJECT_ID}" \
  --description="Number of prediction errors from /api/predict" \
  --log-filter="${LOG_FILTER_BASE} jsonPayload.event=\"prediction_error\""

echo "  [OK] prediction_error_count"

echo ""
echo "=== Injecting project ID into dashboard.json ==="

sed "s/YOUR_PROJECT_ID/${GCP_PROJECT_ID}/g; s/YOUR_CLOUD_RUN_SERVICE/${CLOUD_RUN_SERVICE}/g" \
  monitoring/dashboard.json > /tmp/dashboard_rendered.json

echo "  [OK] dashboard rendered to /tmp/dashboard_rendered.json"

echo ""
echo "=== Creating monitoring dashboard ==="

gcloud monitoring dashboards create \
  --project="${GCP_PROJECT_ID}" \
  --config-from-file=/tmp/dashboard_rendered.json

echo "  [OK] Dashboard created."
echo ""
echo "Open: https://console.cloud.google.com/monitoring/dashboards?project=${GCP_PROJECT_ID}"
