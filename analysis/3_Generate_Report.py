from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "results" / "paired_strategy"

FIGURE_FILES = [
    ("fig_boxplot_boarding_time.png", "Boarding time by strategy"),
    ("fig_hist_paired_differences.png", "Histogram of paired differences with fitted normal density"),
    ("fig_qq_paired_differences.png", "Q-Q plot of paired differences"),
]


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000 or (value != 0 and abs(value) < 0.001):
            return f"{value:.3e}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _load_report_inputs(output_dir: Path) -> dict:
    return {
        "config": _load_json(output_dir / "study_config_snapshot.json"),
        "replication_summary": _load_json(output_dir / "required_replications.json"),
        "strategy_summary": _load_csv(output_dir / "strategy_descriptive_summary.csv"),
        "inferential_summary": _load_csv(output_dir / "paired_inferential_summary.csv"),
        "paired_runs": _load_csv(output_dir / "paired_runs_pairs.csv"),
        "run_failures": _load_csv(output_dir / "run_failures.csv"),
    }


def _build_summary_notes(inputs: dict) -> List[str]:
    infer_df = inputs["inferential_summary"]
    strategy_df = inputs["strategy_summary"]
    run_failures = inputs["run_failures"]
    paired_runs = inputs["paired_runs"]

    notes: List[str] = []
    if not infer_df.empty:
        record = infer_df.iloc[0]
        notes.append(
            f"Selected test: {record.get('selected_test', '')}, p-value: {_format_value(record.get('p_value', ''))}, "
            f"mean paired difference: {_format_value(record.get('mean_paired_difference', ''))} s."
        )
    if not strategy_df.empty:
        best = strategy_df.sort_values("mean", ascending=True).iloc[0]
        notes.append(
            f"Best mean boarding time: {best['strategy']} at {_format_value(best['mean'])} s."
        )
    notes.append(f"Completed pairs: {int(paired_runs['pair_completed'].sum()) if not paired_runs.empty else 0}.")
    notes.append(f"Failed runs: {len(run_failures)}.")
    return notes


def _build_figure_explanations_markdown() -> List[str]:
    return [
        "### Boarding time by strategy",
        "- What it shows: distribution of total boarding times per strategy (median, spread, and outliers).",
        "- How to read it: lower boxes/medians mean faster boarding; narrower spread means more consistency.",
        "- Conclusion: compare central tendency and spread to assess speed and reliability tradeoffs.",
        "",
        "### Histogram of paired differences with fitted normal density",
        "- What it shows: frequency distribution of paired differences (zonal - pyramid) with a fitted normal curve overlay.",
        "- How to read it: values above 0 indicate pyramid is faster; the red curve is a visual normal-reference guide.",
        "- Conclusion: center and spread indicate average gain and variability; compare bars vs curve for rough normality fit.",
        "",
        "### Q-Q plot of paired differences",
        "- What it shows: observed quantiles of paired differences against theoretical normal quantiles.",
        "- How to read it: points close to a straight line suggest approximate normality; systematic bends indicate departures.",
        "- Conclusion: supports whether the paired t-test normality assumption is reasonable.",
        "",
    ]


def generate_markdown_report(output_dir: Path) -> str:
    inputs = _load_report_inputs(output_dir)
    config = inputs["config"]
    replication_summary = inputs["replication_summary"]
    strategy_df = inputs["strategy_summary"]
    infer_df = inputs["inferential_summary"]
    paired_df = inputs["paired_runs"]
    failures_df = inputs["run_failures"]

    title = "Paired Boarding Strategy Report"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fixed_assumptions = config.get("study_context", {}).get("fixed_assumptions", [])

    md: List[str] = [f"# {title}", "", f"Generated: {generated_at}", ""]

    md.append("## Study Design")
    md.append("- Independent variable: boarding strategy")
    md.append("- Primary dependent variable: total boarding time")
    md.append("- Strategies: back-to-front zonal vs modified reverse pyramid")
    if fixed_assumptions:
        md.append("- Fixed assumptions: " + ", ".join(str(item) for item in fixed_assumptions))
    md.append("")

    md.append("## Run Summary")
    md.append(_df_to_markdown_table(pd.DataFrame([{
        "required_replications": replication_summary.get("required_replications", config.get("replications", "")),
        "replications_attempted": replication_summary.get("replications_attempted", int(len(paired_df))),
        "completed_pairs": replication_summary.get("completed_pairs", int(paired_df["pair_completed"].sum()) if not paired_df.empty else 0),
        "master_seed": replication_summary.get("master_seed", config.get("master_seed", "")),
        "load_factor": replication_summary.get("load_factor", config.get("load_factor", "")),
        "luggage_probability": replication_summary.get("luggage_probability", config.get("luggage_probability", "")),
        "cross_zone_violation_rate": replication_summary.get("cross_zone_violation_rate", config.get("cross_zone_violation_rate", "")),
        "paired_runs": int(len(paired_df)),
        "failed_runs": len(failures_df),
    }])))
    md.append("")

    md.append("## Descriptive Statistics")
    md.append(_df_to_markdown_table(strategy_df))
    md.append("")

    md.append("## Paired Inference")
    md.append(_df_to_markdown_table(infer_df))
    md.append("")

    md.append("## Paired Metrics")
    paired_columns = [
        "replication_id",
        "boarding_time_zonal",
        "boarding_time_pyramid",
        "difference",
        "ratio",
        "relative_improvement",
    ]
    if not paired_df.empty:
        md.append(_df_to_markdown_table(paired_df[paired_columns].head(12)))
    else:
        md.append("_No paired results available._")
    md.append("")

    md.append("## Figures")
    for filename, caption in FIGURE_FILES:
        figure_path = output_dir / filename
        if figure_path.exists():
            md.append(f"![{caption}]({filename})")
            md.append("")

    md.extend(_build_figure_explanations_markdown())

    md.append("## Notes")
    if replication_summary:
        md.append(
            "- Replication summary: "
            f"required={_format_value(replication_summary.get('required_replications', ''))}, "
            f"attempted={_format_value(replication_summary.get('replications_attempted', ''))}, "
            f"completed_pairs={_format_value(replication_summary.get('completed_pairs', ''))}."
        )
    for note in _build_summary_notes(inputs):
        md.append(f"- {note}")
    md.append("")

    if not failures_df.empty:
        md.append("## Failures")
        md.append(_df_to_markdown_table(failures_df[["replication_id", "scenario_seed", "strategy", "error_message"]].head(20)))
        md.append("")

    return "\n".join(md).strip() + "\n"


def write_paired_strategy_report(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_text = generate_markdown_report(output_dir)

    markdown_path = output_dir / "paired_strategy_report.md"

    markdown_path.write_text(markdown_text, encoding="utf-8")
    return markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3 (optional): generate markdown report from analysis outputs."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    inferential_path = output_dir / "paired_inferential_summary.csv"
    descriptive_path = output_dir / "strategy_descriptive_summary.csv"

    if not inferential_path.exists() or not descriptive_path.exists():
        raise FileNotFoundError(
            "Missing step-2 outputs paired_inferential_summary.csv or "
            "strategy_descriptive_summary.csv. Run 2_Analyze_Paired_Results.py first."
        )

    markdown_path = write_paired_strategy_report(output_dir)

    print("Step 3 complete: report outputs generated.")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
