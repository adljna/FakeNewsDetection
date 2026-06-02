#!/usr/bin/env bash

set -e

if [ -z "$SERVICE_URL" ]; then
  echo "SERVICE_URL is required."
  echo "Example:"
  echo "SERVICE_URL=https://your-cloud-run-url.run.app ./scripts/smoke_test.sh"
  exit 1
fi

echo "Testing /health endpoint..."
curl -fsS "$SERVICE_URL/health"
echo ""

echo "Testing /api/predict endpoint..."
curl -fsS -X POST "$SERVICE_URL/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a sample news article for fake news detection."}'
echo ""

echo "Smoke test completed successfully."
