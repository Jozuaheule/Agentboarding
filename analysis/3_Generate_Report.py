from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "results" / "paired_strategy"

FIGURE_FILES = [
    ("fig_boxplot_boarding_time.png", "Boarding time by strategy"),
    ("fig_hist_paired_differences.png", "Histogram of paired differences"),
    ("fig_qq_paired_differences.png", "Q-Q plot of paired differences"),
    ("fig_relative_improvement_by_replication.png", "Relative improvement by replication"),
    ("fig_ci_cv_stabilization.png", "CI and CV stabilization across completed pairs"),
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


def _df_to_html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data available.</p>"

    escaped_columns = [html.escape(str(column)) for column in df.columns]
    header = "<tr>" + "".join(f"<th>{column}</th>" for column in escaped_columns) + "</tr>"
    body_rows: List[str] = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{html.escape(_format_value(row[column]))}</td>" for column in df.columns)
        body_rows.append(f"<tr>{cells}</tr>")

    return "<table>" + header + "".join(body_rows) + "</table>"


def _load_report_inputs(output_dir: Path) -> dict:
    return {
        "config": _load_json(output_dir / "study_config_snapshot.json"),
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


def generate_markdown_report(output_dir: Path) -> str:
    inputs = _load_report_inputs(output_dir)
    config = inputs["config"]
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
        "replications": config.get("replications", ""),
        "master_seed": config.get("master_seed", ""),
        "load_factor": config.get("load_factor", ""),
        "luggage_probability": config.get("luggage_probability", ""),
        "cross_zone_violation_rate": config.get("cross_zone_violation_rate", ""),
        "paired_runs": int(len(paired_df)),
        "completed_pairs": int(paired_df["pair_completed"].sum()) if not paired_df.empty else 0,
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

    md.append("## Notes")
    for note in _build_summary_notes(inputs):
        md.append(f"- {note}")
    md.append("")

    if not failures_df.empty:
        md.append("## Failures")
        md.append(_df_to_markdown_table(failures_df[["replication_id", "scenario_seed", "strategy", "error_message"]].head(20)))
        md.append("")

    return "\n".join(md).strip() + "\n"


def generate_html_report(output_dir: Path) -> str:
    inputs = _load_report_inputs(output_dir)
    config = inputs["config"]
    strategy_df = inputs["strategy_summary"]
    infer_df = inputs["inferential_summary"]
    paired_df = inputs["paired_runs"]
    failures_df = inputs["run_failures"]

    title = "Paired Boarding Strategy Report"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fixed_assumptions = config.get("study_context", {}).get("fixed_assumptions", [])

    html_parts: List[str] = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body { font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px auto; max-width: 1100px; line-height: 1.5; color: #1f2937; padding: 0 20px; }",
        "h1, h2 { color: #111827; }",
        "table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }",
        "th, td { border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; vertical-align: top; }",
        "th { background: #f3f4f6; }",
        ".note { background: #f9fafb; border-left: 4px solid #60a5fa; padding: 12px 16px; margin: 16px 0; }",
        ".figure { margin: 20px 0 28px; }",
        ".figure img { max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 8px; }",
        ".muted { color: #6b7280; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='muted'>Generated: {html.escape(generated_at)}</p>",
        "<h2>Study Design</h2>",
        "<ul>",
        "<li>Independent variable: boarding strategy</li>",
        "<li>Primary dependent variable: total boarding time</li>",
        "<li>Strategies: back-to-front zonal vs modified reverse pyramid</li>",
        *[f"<li>{html.escape(str(item))}</li>" for item in fixed_assumptions],
        "</ul>",
        "<h2>Run Summary</h2>",
        _df_to_html_table(pd.DataFrame([{
            "replications": config.get("replications", ""),
            "master_seed": config.get("master_seed", ""),
            "load_factor": config.get("load_factor", ""),
            "luggage_probability": config.get("luggage_probability", ""),
            "cross_zone_violation_rate": config.get("cross_zone_violation_rate", ""),
            "paired_runs": int(len(paired_df)),
            "completed_pairs": int(paired_df["pair_completed"].sum()) if not paired_df.empty else 0,
            "failed_runs": len(failures_df),
        }])),
        "<h2>Descriptive Statistics</h2>",
        _df_to_html_table(strategy_df),
        "<h2>Paired Inference</h2>",
        _df_to_html_table(infer_df),
        "<h2>Paired Metrics</h2>",
        _df_to_html_table(paired_df[["replication_id", "boarding_time_zonal", "boarding_time_pyramid", "difference", "ratio", "relative_improvement"]].head(12)) if not paired_df.empty else "<p>No paired results available.</p>",
        "<h2>Figures</h2>",
    ]

    for filename, caption in FIGURE_FILES:
        figure_path = output_dir / filename
        if figure_path.exists():
            html_parts.extend([
                "<div class='figure'>",
                f"<h3>{html.escape(caption)}</h3>",
                f"<img src='{html.escape(filename)}' alt='{html.escape(caption)}'>",
                "</div>",
            ])

    html_parts.append("<h2>Notes</h2>")
    html_parts.append("<div class='note'>")
    for note in _build_summary_notes(inputs):
        html_parts.append(f"<div>{html.escape(note)}</div>")
    html_parts.append("</div>")

    if not failures_df.empty:
        html_parts.extend([
            "<h2>Failures</h2>",
            _df_to_html_table(failures_df[["replication_id", "scenario_seed", "strategy", "error_message"]].head(20)),
        ])

    html_parts.extend(["</body>", "</html>"])
    return "\n".join(html_parts)


def write_paired_strategy_report(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_text = generate_markdown_report(output_dir)
    html_text = generate_html_report(output_dir)

    markdown_path = output_dir / "paired_strategy_report.md"
    html_path = output_dir / "paired_strategy_report.html"

    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    return markdown_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3 (optional): generate markdown/html reports from analysis outputs."
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

    markdown_path, html_path = write_paired_strategy_report(output_dir)

    print("Step 3 complete: report outputs generated.")
    print(f"Markdown report: {markdown_path}")
    print(f"HTML report: {html_path}")


if __name__ == "__main__":
    main()
