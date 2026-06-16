import json
import os
import sys

metrics_path = "reports/metrics.json"
min_f1 = float(os.getenv("MIN_F1_SCORE"))

with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

f1 = float(metrics.get("f1_weighted", 0))

print(f"F1 score: {f1:.4f}")
print(f"Minimum required F1 score: {min_f1:.4f}")

with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
    summary.write("## Model Metric Gate\n\n")
    summary.write(f"- F1 score: `{f1:.4f}`\n")
    summary.write(f"- Minimum required F1 score: `{min_f1:.4f}`\n")

if f1 < min_f1:
    print("Model failed the metric gate.")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
        summary.write("- Status: Failed\n")
    sys.exit(1)

print("Model passed the metric gate.")
with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
    summary.write("- Status: Passed\n") 