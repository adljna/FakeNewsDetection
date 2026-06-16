import json
import os
import sys


def main():
    with open("best_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    best_f1 = float(metrics.get("f1_weighted", 0))
    accuracy = metrics.get("accuracy", "unknown")

    print(f"Best candidate F1: {best_f1:.4f}")
    print(f"Best candidate accuracy: {accuracy}")

    min_f1_env = os.getenv("MIN_F1_SCORE")
    if min_f1_env:
        min_f1 = float(min_f1_env)
        print(f"Minimum required F1: {min_f1:.4f}")
        if best_f1 < min_f1:
            print(
                f"Best candidate F1 {best_f1:.4f} is below minimum {min_f1:.4f}. "
                "Aborting deployment."
            )
            sys.exit(1)
        print("Model passed the minimum F1 gate.")

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


if __name__ == "__main__":
    main()
