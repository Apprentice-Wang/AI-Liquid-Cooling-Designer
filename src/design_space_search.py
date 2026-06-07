from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Support direct execution from the project root:
#   python src/design_space_search.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import calculate_single_case, load_case


CASE_PATH = PROJECT_ROOT / "examples" / "single_chip_case.yaml"
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figures"

RANDOM_SEED = 42
RANDOM_DESIGN_COUNT = 5000
CHANNEL_SPACING_MM = 0.5


def _prepare_output_dirs():
    """Create output folders before writing CSV files and figures."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _required_width_mm(channel_number, channel_width_mm):
    """Return the width occupied by channels and the fixed inter-channel gaps."""
    return (
        channel_number * channel_width_mm
        + (channel_number - 1) * CHANNEL_SPACING_MM
    )


def _evaluate_design(
    baseline_case,
    flow_rate_l_min,
    channel_number,
    channel_width_mm,
    channel_height_mm,
):
    """Evaluate one geometrically valid design with the model from main.py."""
    case = deepcopy(baseline_case)
    case["coolant"]["volume_flow_rate_L_min"] = float(flow_rate_l_min)
    case["channels"]["number"] = int(channel_number)
    case["channels"]["width_mm"] = float(channel_width_mm)
    case["channels"]["height_mm"] = float(channel_height_mm)

    result = calculate_single_case(case)
    targets = baseline_case["targets"]
    thermal_ok = result["max_temperature_C"] <= targets["max_temperature_C"]
    pressure_ok = result["pressure_drop_kPa"] <= targets["max_pressure_drop_kPa"]

    return {
        "flow_rate_L_min": float(flow_rate_l_min),
        "channel_number": int(channel_number),
        "channel_width_mm": float(channel_width_mm),
        "channel_height_mm": float(channel_height_mm),
        "required_width_mm": _required_width_mm(channel_number, channel_width_mm),
        "max_temperature_C": result["max_temperature_C"],
        "pressure_drop_kPa": result["pressure_drop_kPa"],
        "total_thermal_resistance_K_W": result["total_thermal_resistance_K_W"],
        "pumping_power_W": result["pumping_power_W"],
        "Re": result["Re"],
        "flow_regime": result["flow_regime"],
        "thermal_ok": thermal_ok,
        "pressure_ok": pressure_ok,
        "feasible": thermal_ok and pressure_ok,
    }


def generate_design_space(
    baseline_case,
    random_design_count=RANDOM_DESIGN_COUNT,
    random_seed=RANDOM_SEED,
):
    """
    Generate random designs and calculate only geometrically valid candidates.

    Returning the generated count separately keeps the CSV focused on designs
    that can physically fit within the available cold-plate width.
    """
    rng = np.random.default_rng(random_seed)
    available_width_mm = baseline_case["chip"]["width_mm"]
    records = []

    for _ in range(random_design_count):
        flow_rate_l_min = rng.uniform(0.5, 5.0)
        channel_number = rng.integers(4, 31)
        channel_width_mm = rng.uniform(0.5, 3.0)
        channel_height_mm = rng.uniform(0.5, 4.0)
        required_width_mm = _required_width_mm(channel_number, channel_width_mm)

        # Skip designs that cannot fit across the chip width.
        if required_width_mm > available_width_mm:
            continue

        records.append(
            _evaluate_design(
                baseline_case,
                flow_rate_l_min,
                channel_number,
                channel_width_mm,
                channel_height_mm,
            )
        )

    return pd.DataFrame(records), random_design_count


def _min_max_normalize(series):
    """Normalize a metric while handling a constant-valued column safely."""
    metric_range = series.max() - series.min()
    if metric_range == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / metric_range


def _calculate_balanced_scores(feasible_df):
    """Calculate the weighted thermal-hydraulic score for feasible designs."""
    scored_df = feasible_df.copy()
    normalized_temperature = _min_max_normalize(scored_df["max_temperature_C"])
    normalized_pressure = _min_max_normalize(scored_df["pressure_drop_kPa"])
    normalized_pumping_power = _min_max_normalize(scored_df["pumping_power_W"])
    scored_df["score"] = (
        0.5 * normalized_temperature
        + 0.3 * normalized_pressure
        + 0.2 * normalized_pumping_power
    )
    return scored_df


def _baseline_design(baseline_case):
    """Evaluate the YAML baseline so it can be compared with searched designs."""
    return _evaluate_design(
        baseline_case,
        baseline_case["coolant"]["volume_flow_rate_L_min"],
        baseline_case["channels"]["number"],
        baseline_case["channels"]["width_mm"],
        baseline_case["channels"]["height_mm"],
    )


def save_top_designs(baseline_case, feasible_df):
    """Save baseline, best thermal, best hydraulic, and best balanced designs."""
    if feasible_df.empty:
        raise ValueError("No feasible designs were found. Cannot select top designs.")

    scored_df = _calculate_balanced_scores(feasible_df)
    selections = [
        ("baseline_design", pd.Series(_baseline_design(baseline_case))),
        (
            "best_thermal_design",
            scored_df.loc[scored_df["max_temperature_C"].idxmin()],
        ),
        (
            "best_hydraulic_design",
            scored_df.loc[scored_df["pressure_drop_kPa"].idxmin()],
        ),
        ("best_balanced_design", scored_df.loc[scored_df["score"].idxmin()]),
    ]

    records = []
    for design_type, design in selections:
        record = design.to_dict()
        record["design_type"] = design_type
        records.append(record)

    top_designs_df = pd.DataFrame(records)
    columns = ["design_type"] + [
        column for column in top_designs_df.columns if column != "design_type"
    ]
    top_designs_df = top_designs_df[columns]
    top_designs_df.to_csv(DATA_DIR / "top_designs.csv", index=False)
    return top_designs_df


def _pareto_front(feasible_df):
    """Find feasible designs not dominated in both temperature and pressure."""
    sorted_df = feasible_df.sort_values(
        ["max_temperature_C", "pressure_drop_kPa"]
    )
    pareto_rows = []
    best_pressure_drop = np.inf

    for _, row in sorted_df.iterrows():
        if row["pressure_drop_kPa"] < best_pressure_drop:
            pareto_rows.append(row)
            best_pressure_drop = row["pressure_drop_kPa"]

    return pd.DataFrame(pareto_rows).sort_values("pressure_drop_kPa")


def plot_design_space(results_df, feasible_df):
    """Generate design-space, Pareto-front, and feasible-map figures."""
    infeasible_df = results_df[~results_df["feasible"]]

    plt.figure(figsize=(7, 5))
    plt.scatter(
        infeasible_df["pressure_drop_kPa"],
        infeasible_df["max_temperature_C"],
        s=14,
        alpha=0.35,
        color="tab:gray",
        label="Infeasible",
    )
    plt.scatter(
        feasible_df["pressure_drop_kPa"],
        feasible_df["max_temperature_C"],
        s=18,
        alpha=0.65,
        color="tab:green",
        label="Feasible",
    )
    plt.xlabel("Pressure drop / kPa")
    plt.ylabel("Maximum temperature / degC")
    plt.title("Design space: thermal-hydraulic performance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "design_space_temperature_pressure.png", dpi=300)
    plt.close()

    pareto_df = _pareto_front(feasible_df)
    plt.figure(figsize=(7, 5))
    plt.scatter(
        feasible_df["pressure_drop_kPa"],
        feasible_df["max_temperature_C"],
        s=14,
        alpha=0.3,
        color="tab:green",
        label="Feasible designs",
    )
    plt.plot(
        pareto_df["pressure_drop_kPa"],
        pareto_df["max_temperature_C"],
        color="tab:red",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="Pareto front",
    )
    plt.xlabel("Pressure drop / kPa")
    plt.ylabel("Maximum temperature / degC")
    plt.title("Pareto front: maximum temperature vs pressure drop")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "pareto_front.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(
        feasible_df["channel_number"],
        feasible_df["channel_height_mm"],
        c=feasible_df["max_temperature_C"],
        cmap="viridis",
        s=24,
        alpha=0.7,
    )
    plt.colorbar(label="Maximum temperature / degC")
    plt.xlabel("Channel number")
    plt.ylabel("Channel height / mm")
    plt.title("Feasible design distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feasible_design_map.png", dpi=300)
    plt.close()


def _print_design(label, design):
    """Print the most useful engineering values for one selected design."""
    print(
        f"{label}: flow={design['flow_rate_L_min']:.3f} L/min, "
        f"channels={int(design['channel_number'])}, "
        f"width={design['channel_width_mm']:.3f} mm, "
        f"height={design['channel_height_mm']:.3f} mm, "
        f"Tmax={design['max_temperature_C']:.3f} degC, "
        f"pressure_drop={design['pressure_drop_kPa']:.3f} kPa, "
        f"pumping_power={design['pumping_power_W']:.4f} W"
    )


def main():
    """Run the random multi-parameter cold-plate design-space search."""
    _prepare_output_dirs()
    baseline_case = load_case(CASE_PATH)
    results_df, generated_count = generate_design_space(baseline_case)
    feasible_df = results_df[results_df["feasible"]].copy()

    results_df.to_csv(DATA_DIR / "design_space_results.csv", index=False)
    feasible_df.to_csv(DATA_DIR / "feasible_designs.csv", index=False)
    top_designs_df = save_top_designs(baseline_case, feasible_df)
    plot_design_space(results_df, feasible_df)

    feasible_ratio = len(feasible_df) / len(results_df) if len(results_df) else 0.0
    print("\n========== Design Space Search Summary ==========")
    print(f"Random designs generated: {generated_count}")
    print(f"Designs passing geometric constraint: {len(results_df)}")
    print(f"Feasible designs: {len(feasible_df)}")
    print(f"Feasible ratio among geometrically valid designs: {feasible_ratio:.2%}")

    indexed_top_designs = top_designs_df.set_index("design_type")
    _print_design(
        "Lowest temperature design",
        indexed_top_designs.loc["best_thermal_design"],
    )
    _print_design(
        "Lowest pressure-drop design",
        indexed_top_designs.loc["best_hydraulic_design"],
    )
    _print_design(
        "Best balanced design",
        indexed_top_designs.loc["best_balanced_design"],
    )

    print("\nGenerated files:")
    print("1. data/design_space_results.csv")
    print("2. data/feasible_designs.csv")
    print("3. data/top_designs.csv")
    print("4. figures/design_space_temperature_pressure.png")
    print("5. figures/pareto_front.png")
    print("6. figures/feasible_design_map.png")


if __name__ == "__main__":
    main()
