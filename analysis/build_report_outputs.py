from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.paired_strategy_core import summarize_by_strategy
from analysis.paired_strategy_report import write_paired_strategy_report
from analysis.run_paired_ttest import compute_paired_ttest_summary
from analysis.study_config import DEFAULT_OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Build descriptive, inferential, and report outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    runs_path = output_dir / "paired_runs_long.csv"
    pairs_path = output_dir / "paired_runs_pairs.csv"

    if not runs_path.exists() or not pairs_path.exists():
        raise FileNotFoundError("Missing paired run outputs. Run run_paired_replications.py first.")

    runs_df = pd.read_csv(runs_path)
    pairs_df = pd.read_csv(pairs_path)

    summary_df = summarize_by_strategy(runs_df)
    summary_df.to_csv(output_dir / "strategy_descriptive_summary.csv", index=False)

    infer_df = compute_paired_ttest_summary(pairs_df)
    infer_df.to_csv(output_dir / "paired_inferential_summary.csv", index=False)

    failures_df = runs_df[~runs_df["completed"].astype(bool)].copy()
    failures_df.to_csv(output_dir / "run_failures.csv", index=False)

    markdown_path, html_path = write_paired_strategy_report(output_dir)

    print("Report outputs built.")
    print(f"Markdown report: {markdown_path}")
    print(f"HTML report: {html_path}")


if __name__ == "__main__":
    main()
