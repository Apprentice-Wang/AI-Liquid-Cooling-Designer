from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Allow this file to be run as either:
#   python src/sensitivity_analysis.py
# or:
#   python -m src.sensitivity_analysis
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import calculate_single_case, load_case


CASE_PATH = PROJECT_ROOT / "examples" / "single_chip_case.yaml"
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figures"


def _prepare_output_dirs():
    """Create output folders before writing CSV files and figures."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _base_case():
    """Load the baseline case used by all sensitivity studies."""
    return load_case(CASE_PATH)


def _calculate_records(base_case, parameter_values, update_case, parameter_name):
    """
    Run one sensitivity sweep.

    update_case is a small callback that changes only the scanned parameter,
    leaving all physics and calculation formulas inside main.py untouched.
    """
    records = []
    targets = base_case["targets"]
    max_temperature_target = targets["max_temperature_C"]
    max_pressure_drop_target = targets["max_pressure_drop_kPa"]

    for value in parameter_values:
        case = deepcopy(base_case)
        update_case(case, value)
        result = calculate_single_case(case)
        thermal_ok = result["max_temperature_C"] <= max_temperature_target
        pressure_ok = result["pressure_drop_kPa"] <= max_pressure_drop_target

        records.append(
            {
                parameter_name: value,
                "max_temperature_C": result["max_temperature_C"],
                "pressure_drop_kPa": result["pressure_drop_kPa"],
                "thermal_resistance_K_W": result["total_thermal_resistance_K_W"],
                "pumping_power_W": result["pumping_power_W"],
                "Re": result["Re"],
                "flow_regime": result["flow_regime"],
                "thermal_ok": thermal_ok,
                "pressure_ok": pressure_ok,
                "feasible": thermal_ok and pressure_ok,
            }
        )

    return pd.DataFrame(records)


def _plot_dual_axis(
    df,
    x_column,
    x_label,
    title,
    output_path,
):
    """Plot max temperature and pressure drop on one chart with two y axes."""
    fig, ax_temp = plt.subplots(figsize=(6.5, 4.2))

    temp_line = ax_temp.plot(
        df[x_column],
        df["max_temperature_C"],
        color="tab:red",
        marker="o",
        label="Max temperature",
    )
    ax_temp.set_xlabel(x_label)
    ax_temp.set_ylabel("Maximum temperature / degC", color="tab:red")
    ax_temp.tick_params(axis="y", labelcolor="tab:red")
    ax_temp.grid(True, alpha=0.3)

    ax_pressure = ax_temp.twinx()
    pressure_line = ax_pressure.plot(
        df[x_column],
        df["pressure_drop_kPa"],
        color="tab:blue",
        marker="s",
        label="Pressure drop",
    )
    ax_pressure.set_ylabel("Pressure drop / kPa", color="tab:blue")
    ax_pressure.tick_params(axis="y", labelcolor="tab:blue")

    lines = temp_line + pressure_line
    labels = [line.get_label() for line in lines]
    ax_temp.legend(lines, labels, loc="best")
    ax_temp.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_power_sensitivity():
    """
    Sweep chip power and save:
    - data/power_sensitivity.csv
    - figures/power_vs_max_temperature.png
    """
    _prepare_output_dirs()
    base_case = _base_case()

    powers = np.linspace(200.0, 1000.0, 17)

    def update_power(case, value):
        case["chip"]["power_W"] = float(value)

    df = _calculate_records(base_case, powers, update_power, "chip_power_W")
    df.to_csv(DATA_DIR / "power_sensitivity.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(df["chip_power_W"], df["max_temperature_C"], marker="o")
    plt.xlabel("Chip power / W")
    plt.ylabel("Maximum temperature / degC")
    plt.title("Effect of chip power on maximum temperature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "power_vs_max_temperature.png", dpi=300)
    plt.close()

    return df


def run_channel_number_sensitivity():
    """
    Sweep channel number and save:
    - data/channel_number_sensitivity.csv
    - figures/channel_number_sensitivity.png
    """
    _prepare_output_dirs()
    base_case = _base_case()

    channel_numbers = np.arange(4, 26, 2, dtype=int)

    def update_channel_number(case, value):
        case["channels"]["number"] = int(value)

    df = _calculate_records(
        base_case,
        channel_numbers,
        update_channel_number,
        "channel_number",
    )
    df.to_csv(DATA_DIR / "channel_number_sensitivity.csv", index=False)

    _plot_dual_axis(
        df=df,
        x_column="channel_number",
        x_label="Channel number",
        title="Effect of channel number",
        output_path=FIGURE_DIR / "channel_number_sensitivity.png",
    )

    return df


def run_channel_height_sensitivity():
    """
    Sweep channel height and save:
    - data/channel_height_sensitivity.csv
    - figures/channel_height_sensitivity.png
    """
    _prepare_output_dirs()
    base_case = _base_case()

    heights = np.linspace(0.5, 4.0, 15)

    def update_channel_height(case, value):
        case["channels"]["height_mm"] = float(value)

    df = _calculate_records(
        base_case,
        heights,
        update_channel_height,
        "channel_height_mm",
    )
    df.to_csv(DATA_DIR / "channel_height_sensitivity.csv", index=False)

    _plot_dual_axis(
        df=df,
        x_column="channel_height_mm",
        x_label="Channel height / mm",
        title="Effect of channel height",
        output_path=FIGURE_DIR / "channel_height_sensitivity.png",
    )

    return df


def summarize_feasible_designs():
    """
    Summarize feasible designs and the thermal/hydraulic extremes for each scan.

    The summary stores each selected design's scanned parameter value so it can
    be traced back to the corresponding detailed sensitivity CSV.
    """
    _prepare_output_dirs()
    studies = [
        ("Power sensitivity", "power_sensitivity.csv", "chip_power_W"),
        (
            "Channel number sensitivity",
            "channel_number_sensitivity.csv",
            "channel_number",
        ),
        (
            "Channel height sensitivity",
            "channel_height_sensitivity.csv",
            "channel_height_mm",
        ),
    ]
    summary_records = []

    for study_name, csv_name, parameter_name in studies:
        df = pd.read_csv(DATA_DIR / csv_name)
        feasible_count = int(df["feasible"].sum())
        lowest_temperature_design = df.loc[df["max_temperature_C"].idxmin()]
        lowest_pressure_design = df.loc[df["pressure_drop_kPa"].idxmin()]

        summary_records.append(
            {
                "sensitivity": study_name,
                "total_design_count": len(df),
                "feasible_design_count": feasible_count,
                "lowest_temperature_parameter": parameter_name,
                "lowest_temperature_parameter_value": lowest_temperature_design[
                    parameter_name
                ],
                "lowest_temperature_C": lowest_temperature_design[
                    "max_temperature_C"
                ],
                "lowest_temperature_pressure_drop_kPa": lowest_temperature_design[
                    "pressure_drop_kPa"
                ],
                "lowest_pressure_parameter": parameter_name,
                "lowest_pressure_parameter_value": lowest_pressure_design[
                    parameter_name
                ],
                "lowest_pressure_drop_kPa": lowest_pressure_design[
                    "pressure_drop_kPa"
                ],
                "lowest_pressure_max_temperature_C": lowest_pressure_design[
                    "max_temperature_C"
                ],
            }
        )

        print(
            f"{study_name}: {feasible_count}/{len(df)} designs satisfy "
            "temperature and pressure-drop constraints."
        )
        print(
            "  Lowest temperature design: "
            f"{parameter_name}={lowest_temperature_design[parameter_name]}, "
            f"max_temperature_C={lowest_temperature_design['max_temperature_C']:.3f}, "
            f"pressure_drop_kPa={lowest_temperature_design['pressure_drop_kPa']:.3f}"
        )
        print(
            "  Lowest pressure-drop design: "
            f"{parameter_name}={lowest_pressure_design[parameter_name]}, "
            f"max_temperature_C={lowest_pressure_design['max_temperature_C']:.3f}, "
            f"pressure_drop_kPa={lowest_pressure_design['pressure_drop_kPa']:.3f}"
        )

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(DATA_DIR / "sensitivity_summary.csv", index=False)
    return summary_df


def main():
    """Run all sensitivity analyses from one command."""
    run_power_sensitivity()
    run_channel_number_sensitivity()
    run_channel_height_sensitivity()

    print("\nEngineering constraint assessment:")
    summarize_feasible_designs()

    print("Generated sensitivity analysis files:")
    print("1. data/power_sensitivity.csv")
    print("2. data/channel_number_sensitivity.csv")
    print("3. data/channel_height_sensitivity.csv")
    print("4. data/sensitivity_summary.csv")
    print("5. figures/power_vs_max_temperature.png")
    print("6. figures/channel_number_sensitivity.png")
    print("7. figures/channel_height_sensitivity.png")


if __name__ == "__main__":
    main()
