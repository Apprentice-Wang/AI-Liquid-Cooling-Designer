"""Generate Markdown and HTML design reports from project output data.

Run from the project root with:
    python src/report_generator.py
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

CASE_FILE = EXAMPLES_DIR / "single_chip_case.yaml"
RECOMMENDED_FILE = DATA_DIR / "recommended_designs.csv"
COMPARISON_FILE = DATA_DIR / "recommended_designs_v1_vs_v2.csv"
METRICS_FILE = DATA_DIR / "surrogate_model_metrics.csv"

MARKDOWN_REPORT = REPORTS_DIR / "liquid_cooling_design_report.md"
HTML_REPORT = REPORTS_DIR / "liquid_cooling_design_report.html"


def read_yaml(path: Path) -> dict[str, Any] | None:
    """Read a YAML mapping, returning None when the file is unavailable."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)
    return content if isinstance(content, dict) else {}


def read_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV file, returning None when the file is unavailable."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def nested_value(data: dict[str, Any], *keys: str) -> Any:
    """Safely retrieve a value from a nested dictionary."""
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return "N/A"
        value = value[key]
    return value


def format_value(value: Any) -> str:
    """Format report values compactly while preserving useful precision."""
    if pd.isna(value):
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    """Create a Markdown table without requiring the optional tabulate package."""
    header_list = list(headers)
    lines = [
        "| " + " | ".join(header_list) + " |",
        "| " + " | ".join("---" for _ in header_list) + " |",
    ]
    for row in rows:
        cells = [format_value(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_input_section(case_data: dict[str, Any] | None) -> str:
    if case_data is None:
        return "Data file not found."

    rows = [
        ("chip length", nested_value(case_data, "chip", "length_mm"), "mm"),
        ("chip width", nested_value(case_data, "chip", "width_mm"), "mm"),
        ("chip power", nested_value(case_data, "chip", "power_W"), "W"),
        ("coolant type", nested_value(case_data, "coolant", "type"), "-"),
        (
            "inlet temperature",
            nested_value(case_data, "coolant", "inlet_temperature_C"),
            "°C",
        ),
        (
            "target max temperature",
            nested_value(case_data, "targets", "max_temperature_C"),
            "°C",
        ),
        (
            "target max pressure drop",
            nested_value(case_data, "targets", "max_pressure_drop_kPa"),
            "kPa",
        ),
    ]
    return markdown_table(["Parameter", "Value", "Unit"], rows)


def build_metrics_section(metrics: pd.DataFrame | None) -> str:
    if metrics is None:
        return "Data file not found."
    if metrics.empty:
        return "No surrogate model performance data available."

    # Prefer the overall evaluation subset, then choose the highest-R2 model.
    candidates = metrics.copy()
    if "evaluation_subset" in candidates.columns:
        overall = candidates[candidates["evaluation_subset"] == "overall"]
        if not overall.empty:
            candidates = overall

    rows = []
    target_labels = {
        "max_temperature_C": "temperature best model",
        "pressure_drop_kPa": "pressure-drop best model",
    }
    for target, label in target_labels.items():
        target_rows = candidates[candidates.get("target") == target]
        if target_rows.empty:
            rows.append((label, "N/A", "N/A", "N/A", "N/A"))
            continue
        best = target_rows.loc[pd.to_numeric(target_rows["R2"], errors="coerce").idxmax()]
        model_name = best.get("model", "N/A")
        transform = best.get("transform", "N/A")
        if transform not in ("N/A", "raw"):
            model_name = f"{model_name} ({transform})"
        rows.append((label, model_name, best.get("R2"), best.get("MAE"), best.get("RMSE")))

    return markdown_table(["Target", "Best Model", "R2", "MAE", "RMSE"], rows)


def normalized_recommendations(recommended: pd.DataFrame) -> pd.DataFrame:
    """Normalize current and future recommender column names for reporting."""
    normalized = recommended.copy()
    aliases = {
        "recommendation_rank": "rank",
        "engineering_score": "physics_engineering_score",
    }
    for source, destination in aliases.items():
        if destination not in normalized.columns and source in normalized.columns:
            normalized[destination] = normalized[source]
    return normalized


def build_recommendations_section(recommended: pd.DataFrame | None) -> str:
    if recommended is None:
        return "Data file not found."
    if recommended.empty:
        return "No recommended designs available."

    columns = [
        "rank",
        "flow_rate_L_min",
        "channel_number",
        "channel_width_mm",
        "channel_height_mm",
        "physics_max_temperature_C",
        "physics_pressure_drop_kPa",
        "physics_pumping_power_W",
        "physics_Re",
        "physics_flow_regime",
        "physics_engineering_score",
    ]
    normalized = normalized_recommendations(recommended)
    rows = [
        [row.get(column, "N/A") for column in columns]
        for _, row in normalized.head(10).iterrows()
    ]
    return markdown_table(columns, rows)


def get_best_design(recommended: pd.DataFrame | None) -> pd.Series | None:
    if recommended is None or recommended.empty:
        return None
    normalized = normalized_recommendations(recommended)
    if "rank" in normalized.columns:
        rank_one = normalized[pd.to_numeric(normalized["rank"], errors="coerce") == 1]
        if not rank_one.empty:
            return rank_one.iloc[0]
    return normalized.iloc[0]


def build_best_design_section(best: pd.Series | None) -> str:
    if best is None:
        return "Data file not found."

    return (
        "Rank=1 方案在经过 AI 初筛后，已由物理模型完成复核。其主要参数与性能为："
        f"流量 **{format_value(best.get('flow_rate_L_min'))} L/min**，"
        f"通道数量 **{format_value(best.get('channel_number'))}**，"
        f"通道宽度 **{format_value(best.get('channel_width_mm'))} mm**，"
        f"通道高度 **{format_value(best.get('channel_height_mm'))} mm**；"
        f"物理模型计算的最高温度为 **{format_value(best.get('physics_max_temperature_C'))} °C**，"
        f"压降为 **{format_value(best.get('physics_pressure_drop_kPa'))} kPa**，"
        f"泵功为 **{format_value(best.get('physics_pumping_power_W'))} W**，"
        f"最终工程评分为 **{format_value(best.get('physics_engineering_score'))}**。"
    )


def build_comparison_section(comparison: pd.DataFrame | None) -> str:
    if comparison is None:
        return "Data file not found."

    summary = ""
    if not comparison.empty:
        summary = "\n\n" + markdown_table(comparison.columns, comparison.itertuples(index=False, name=None))
    return (
        "- V1 容易产生过冷和高流量推荐方案。\n"
        "- V2 加入过冷惩罚、流量惩罚和制造难度惩罚。\n"
        "- V2 更接近工程约束下的低系统代价设计。"
        + summary
    )


def build_markdown(
    case_data: dict[str, Any] | None,
    recommended: pd.DataFrame | None,
    comparison: pd.DataFrame | None,
    metrics: pd.DataFrame | None,
) -> str:
    """Assemble the complete Markdown report."""
    best = get_best_design(recommended)
    return f"""# AI-Liquid-Cooling-Designer Report

## 1. Design Objective

面向 AI 芯片/功率器件液冷冷板早期热设计，快速筛选满足温度和压降约束的冷板方案。

## 2. Input Conditions

{build_input_section(case_data)}

## 3. Methodology

Physics-based model → Design space search → Surrogate model → AI recommender → Physics verification

## 4. Surrogate Model Performance

{build_metrics_section(metrics)}

## 5. Recommended Designs

{build_recommendations_section(recommended)}

## 6. Best Recommended Design

{build_best_design_section(best)}

## 7. V1 vs V2 Recommendation Comparison

{build_comparison_section(comparison)}

## 8. Engineering Interpretation

推荐方案满足最高温度约束，且压降远低于目标上限。当前推荐结果均经过物理模型复核；AI 预测仅用于快速筛选，不作为最终设计依据。

## 9. Model Limitations

- 当前采用简化热阻模型。
- 未考虑歧管流量分配。
- 未考虑接触热阻。
- 未考虑非均匀热源。
- 未考虑冷却液物性随温度变化。
- 最终设计仍需 CFD 或实验验证。

## 10. Next CFD Validation Plan

后续将选取 rank=1 到 rank=3 的推荐方案，使用 Fluent 或 COMSOL 建立三维冷板模型，验证温度场、压降和流量分布。
"""


def markdown_to_html(markdown_text: str) -> str:
    """Convert the generated report's simple Markdown subset to HTML."""
    lines = markdown_text.splitlines()
    output: list[str] = []
    in_list = False
    index = 0

    while index < len(lines):
        line = lines[index]
        if line.startswith("| ") and index + 1 < len(lines) and lines[index + 1].startswith("| ---"):
            if in_list:
                output.append("</ul>")
                in_list = False
            headers = [cell.strip() for cell in line.strip("|").split("|")]
            output.append("<table><thead><tr>")
            output.extend(f"<th>{html.escape(cell)}</th>" for cell in headers)
            output.append("</tr></thead><tbody>")
            index += 2
            while index < len(lines) and lines[index].startswith("| "):
                cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
                output.append("<tr>")
                output.extend(f"<td>{html.escape(cell)}</td>" for cell in cells)
                output.append("</tr>")
                index += 1
            output.append("</tbody></table>")
            continue
        if line.startswith("- "):
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            if in_list:
                output.append("</ul>")
                in_list = False
            if line.startswith("# "):
                output.append(f"<h1>{html.escape(line[2:])}</h1>")
            elif line.startswith("## "):
                output.append(f"<h2>{html.escape(line[3:])}</h2>")
            elif line.strip():
                escaped = html.escape(line)
                # Preserve the limited bold markup used in the generated report.
                parts = escaped.split("**")
                escaped = "".join(
                    f"<strong>{part}</strong>" if i % 2 else part
                    for i, part in enumerate(parts)
                )
                output.append(f"<p>{escaped}</p>")
        index += 1

    if in_list:
        output.append("</ul>")

    body = "\n".join(output)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI-Liquid-Cooling-Designer Report</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 24px; line-height: 1.6; color: #20242a; }}
    h1, h2 {{ color: #145b8c; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #cbd5df; padding: 8px; text-align: left; }}
    th {{ background: #eaf3f8; }}
    tr:nth-child(even) {{ background: #f7fafc; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    """Load project data and write both report formats."""
    case_data = read_yaml(CASE_FILE)
    recommended = read_csv(RECOMMENDED_FILE)
    comparison = read_csv(COMPARISON_FILE)
    metrics = read_csv(METRICS_FILE)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    markdown_report = build_markdown(case_data, recommended, comparison, metrics)
    MARKDOWN_REPORT.write_text(markdown_report, encoding="utf-8")
    HTML_REPORT.write_text(markdown_to_html(markdown_report), encoding="utf-8")

    recommendation_count = 0 if recommended is None else len(recommended)
    best = get_best_design(recommended)

    print(f"Markdown report saved to: {MARKDOWN_REPORT}")
    print(f"HTML report saved to: {HTML_REPORT}")
    print(f"Recommended design count: {recommendation_count}")
    if best is None:
        print("Rank=1 design: Data file not found.")
    else:
        print(
            "Rank=1 design: "
            f"temperature={format_value(best.get('physics_max_temperature_C'))} °C, "
            f"pressure_drop={format_value(best.get('physics_pressure_drop_kPa'))} kPa, "
            f"flow_rate={format_value(best.get('flow_rate_L_min'))} L/min"
        )


if __name__ == "__main__":
    main()
