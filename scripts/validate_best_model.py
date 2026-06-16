import json
import os


def main():
    with open("best_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    best_f1 = metrics.get("f1_weighted", "unknown")
    accuracy = metrics.get("accuracy", "unknown")

    print(f"Best candidate F1: {best_f1}")
    print(f"Best candidate accuracy: {accuracy}")

    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as env:
            env.write(f"BEST_F1={best_f1}\n")
            env.write(f"BEST_ACCURACY={accuracy}\n")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as summary:
            summary.write("## Production Model Promotion\n\n")
            summary.write(f"- Best candidate F1: `{best_f1}`\n")
            summary.write(f"- Best candidate accuracy: `{accuracy}`\n")