import json
import os
import subprocess
bucket = os.environ["MODEL_BUCKET"]

def main():
    current_metrics_path = "current_metrics.json"
    best_metrics_uri = f"gs://{bucket}/reports/candidate/best/metrics.json"

    current_f1 = 0.0
    previous_best_found = False

    result = subprocess.run(
        ["gcloud", "storage", "cp", best_metrics_uri, current_metrics_path],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        previous_best_found = True
        with open(current_metrics_path, "r", encoding="utf-8") as f:
            current_f1 = float(json.load(f).get("f1_weighted", 0))
    else:
        print("No previous best candidate model found.")
        print("This model can become the first best candidate if it passed the metric gate.")

    with open("reports/metrics.json", "r", encoding="utf-8") as f:
        new_f1 = float(json.load(f).get("f1_weighted", 0))

    is_best = new_f1 > current_f1

    print(f"Previous best found: {previous_best_found}")
    print(f"Current best F1: {current_f1:.4f}")
    print(f"New model F1: {new_f1:.4f}")

    if is_best:
        print("New model is better. It will be promoted to candidate/best.")
    else:
        print("New model is not better. Current best candidate will be kept.")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as out:
            out.write(f"is_best={'true' if is_best else 'false'}\n")
            out.write(f"current_f1={current_f1:.4f}\n")
            out.write(f"new_f1={new_f1:.4f}\n")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as summary:
            summary.write("\n## Candidate Model Comparison\n\n")
            summary.write(f"- Previous best found: `{previous_best_found}`\n")
            summary.write(f"- Current best F1: `{current_f1:.4f}`\n")
            summary.write(f"- New model F1: `{new_f1:.4f}`\n")
            summary.write(f"- Promoted to candidate/best: `{'true' if is_best else 'false'}`\n")
