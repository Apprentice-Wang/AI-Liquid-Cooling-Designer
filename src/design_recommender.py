from copy import deepcopy
from pathlib import Path
import sys

import joblib
import matplotlib

# Save figures reliably when the recommender runs as a batch script.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Support direct execution from the project root:
#   python src/design_recommender.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import calculate_single_case, load_case


CASE_PATH = PROJECT_ROOT / "examples" / "single_chip_case.yaml"
TEMPERATURE_MODEL_PATH = PROJECT_ROOT / "models" / "temperature_model.pkl"
PRESSURE_MODEL_PATH = PROJECT_ROOT / "models" / "pressure_drop_model.pkl"
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figures"

RANDOM_SEED = 42
CANDIDATE_COUNT = 50000
RECOMMENDATION_COUNT = 20
AI_PREFILTER_COUNT = 200
CHANNEL_SPACING_MM = 0.5
SURROGATE_TEMPERATURE_SAFETY_MARGIN_C = 3.0
SURROGATE_PRESSURE_SAFETY_MARGIN_KPA = 1.0
PRESSURE_FLOOR_KPA = 1e-4


def _prepare_output_dirs():
    """Create folders used for recommendation data and figures."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _load_model_bundle(path, expected_target):
    """Load and validate a surrogate-model bundle created by surrogate_model.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path.relative_to(PROJECT_ROOT)}. "
            "Run python src/surrogate_model.py first."
        )

    bundle = joblib.load(path)
    required_keys = {"model", "features", "target", "transform"}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError(
            f"{path.name} is not a metadata bundle. "
            "Run python src/surrogate_model.py to regenerate trained models."
        )
    if bundle["target"] != expected_target:
        raise ValueError(
            f"{path.name} target is {bundle['target']}, expected {expected_target}."
        )
    if bundle["transform"] not in {"none", "log1p"}:
        raise ValueError(f"Unsupported transform in {path.name}: {bundle['transform']}")

    return bundle


def _required_width_mm(channel_number, channel_width_mm):
    """Calculate the cold-plate width occupied by channels and fixed spacing."""
    return (
        channel_number * channel_width_mm
        + (channel_number - 1) * CHANNEL_SPACING_MM
    )


def _generate_candidates(baseline_case, candidate_count=CANDIDATE_COUNT):
    """Generate random designs and retain candidates that pass geometry checks."""
    rng = np.random.default_rng(RANDOM_SEED)
    candidates = pd.DataFrame(
        {
            "flow_rate_L_min": rng.uniform(0.5, 5.0, candidate_count),
            "channel_number": rng.integers(4, 31, candidate_count),
            "channel_width_mm": rng.uniform(0.5, 3.0, candidate_count),
            "channel_height_mm": rng.uniform(0.5, 4.0, candidate_count),
        }
    )
    candidates["required_width_mm"] = _required_width_mm(
        candidates["channel_number"],
        candidates["channel_width_mm"],
    )
    available_width_mm = baseline_case["chip"]["width_mm"]
    valid_candidates = candidates[
        candidates["required_width_mm"] <= available_width_mm
    ].copy()
    return valid_candidates, candidate_count


def _predict(bundle, candidates):
    """Predict with the bundle feature order and restore transformed outputs."""
    predictions = bundle["model"].predict(candidates[bundle["features"]])
    if bundle["transform"] == "log1p":
        predictions = np.expm1(predictions)
    if bundle["target"] == "pressure_drop_kPa":
        # Small negative values can appear near zero after surrogate regression.
        # Keep displayed pressure non-zero to avoid implying zero hydraulic loss.
        predictions = np.maximum(predictions, PRESSURE_FLOOR_KPA)
    return predictions


def _predict_restored_raw(bundle, candidates):
    """
    Predict with inverse target transform but without display flooring.

    This preserves the raw pressure surrogate output after expm1 so users can
    inspect ultra-low pressure behavior separately from the displayed value.
    """
    predictions = bundle["model"].predict(candidates[bundle["features"]])
    if bundle["transform"] == "log1p":
        predictions = np.expm1(predictions)
    return predictions


def _min_max_normalize(series):
    """Normalize one score component while handling constant columns safely."""
    value_range = series.max() - series.min()
    if value_range == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / value_range


def _score_feasible_designs(feasible_df):
    """Calculate the original v1 score for an apples-to-apples comparison."""
    scored_df = feasible_df.copy()
    normalized_temperature = _min_max_normalize(
        scored_df["predicted_max_temperature_C"]
    )
    normalized_pressure = _min_max_normalize(
        scored_df["predicted_pressure_drop_kPa"]
    )
    scored_df["v1_score"] = 0.6 * normalized_temperature + 0.4 * normalized_pressure
    return scored_df


def _calculate_manufacturing_score(feasible_df, available_width_mm):
    """
    Estimate manufacturing difficulty with four transparent penalty increments.

    A lower score indicates a design with wider, taller, fewer, and less densely
    packed channels. The score is intentionally simple for early-stage ranking.
    """
    width_penalty = (feasible_df["channel_width_mm"] < 0.8).astype(int)
    height_penalty = (feasible_df["channel_height_mm"] < 1.0).astype(int)
    channel_count_penalty = (feasible_df["channel_number"] > 24).astype(int)
    packing_penalty = (
        feasible_df["required_width_mm"] / available_width_mm > 0.9
    ).astype(int)
    return width_penalty + height_penalty + channel_count_penalty + packing_penalty


def _score_engineering_designs(feasible_df, available_width_mm, target_temperature):
    """Calculate the v2 target-oriented engineering recommendation score."""
    scored_df = feasible_df.copy()
    scored_df["manufacturing_score"] = _calculate_manufacturing_score(
        scored_df,
        available_width_mm,
    )
    recommended_temperature_lower_bound_c = target_temperature - 25
    scored_df["overcooling_penalty_C"] = np.maximum(
        recommended_temperature_lower_bound_c
        - scored_df["predicted_max_temperature_C"],
        0,
    )
    scored_df["normalized_pressure_drop"] = _min_max_normalize(
        scored_df["predicted_pressure_drop_kPa"]
    )
    scored_df["normalized_flow_rate"] = _min_max_normalize(
        scored_df["flow_rate_L_min"]
    )
    scored_df["normalized_manufacturing_score"] = _min_max_normalize(
        scored_df["manufacturing_score"]
    )
    scored_df["normalized_overcooling_penalty"] = _min_max_normalize(
        scored_df["overcooling_penalty_C"]
    )
    scored_df["engineering_score"] = (
        0.35 * scored_df["normalized_pressure_drop"]
        + 0.25 * scored_df["normalized_flow_rate"]
        + 0.25 * scored_df["normalized_manufacturing_score"]
        + 0.15 * scored_df["normalized_overcooling_penalty"]
    )
    return scored_df


def _score_physics_engineering_designs(verified_df, target_temperature):
    """Re-rank physics-feasible designs using verified performance values."""
    scored_df = verified_df.copy()
    recommended_temperature_lower_bound_c = target_temperature - 25
    scored_df["physics_overcooling_penalty_C"] = np.maximum(
        recommended_temperature_lower_bound_c
        - scored_df["physics_max_temperature_C"],
        0,
    )
    scored_df["normalized_physics_pressure_drop"] = _min_max_normalize(
        scored_df["physics_pressure_drop_kPa"]
    )
    scored_df["normalized_flow_rate"] = _min_max_normalize(
        scored_df["flow_rate_L_min"]
    )
    scored_df["normalized_manufacturing_score"] = _min_max_normalize(
        scored_df["manufacturing_score"]
    )
    scored_df["normalized_physics_overcooling_penalty"] = _min_max_normalize(
        scored_df["physics_overcooling_penalty_C"]
    )
    scored_df["physics_engineering_score"] = (
        0.35 * scored_df["normalized_physics_pressure_drop"]
        + 0.25 * scored_df["normalized_flow_rate"]
        + 0.25 * scored_df["normalized_manufacturing_score"]
        + 0.15 * scored_df["normalized_physics_overcooling_penalty"]
    )
    return scored_df


def _verify_with_physics_model(recommended_df, baseline_case):
    """Recalculate recommended designs with the original engineering model."""
    verified_records = []
    targets = baseline_case["targets"]

    for _, design in recommended_df.iterrows():
        case = deepcopy(baseline_case)
        case["coolant"]["volume_flow_rate_L_min"] = float(design["flow_rate_L_min"])
        case["channels"]["number"] = int(design["channel_number"])
        case["channels"]["width_mm"] = float(design["channel_width_mm"])
        case["channels"]["height_mm"] = float(design["channel_height_mm"])
        result = calculate_single_case(case)
        physics_thermal_ok = (
            result["max_temperature_C"] <= targets["max_temperature_C"]
        )
        physics_pressure_ok = (
            result["pressure_drop_kPa"] <= targets["max_pressure_drop_kPa"]
        )

        record = design.to_dict()
        record.update(
            {
                "physics_max_temperature_C": result["max_temperature_C"],
                "physics_pressure_drop_kPa": result["pressure_drop_kPa"],
                "physics_total_thermal_resistance_K_W": result[
                    "total_thermal_resistance_K_W"
                ],
                "physics_pumping_power_W": result["pumping_power_W"],
                "physics_Re": result["Re"],
                "physics_flow_regime": result["flow_regime"],
                "physics_thermal_ok": physics_thermal_ok,
                "physics_pressure_ok": physics_pressure_ok,
                "physics_feasible": physics_thermal_ok and physics_pressure_ok,
            }
        )
        verified_records.append(record)

    return pd.DataFrame(verified_records)


def _select_physics_verified_recommendations(
    ranked_candidates_df,
    baseline_case,
    recommendation_count=RECOMMENDATION_COUNT,
):
    """
    Verify ranked candidates until enough physics-feasible designs are collected.

    Rejected designs are skipped and replaced with the next ranked candidate.
    """
    selected_records = []
    rejected_count = 0
    verified_count = 0

    for _, candidate in ranked_candidates_df.iterrows():
        verified_df = _verify_with_physics_model(
            candidate.to_frame().T,
            baseline_case,
        )
        verified_design = verified_df.iloc[0].to_dict()
        verified_count += 1

        if verified_design["physics_feasible"]:
            selected_records.append(verified_design)
            if len(selected_records) >= recommendation_count:
                break
        else:
            rejected_count += 1

    selected_df = pd.DataFrame(selected_records)
    if not selected_df.empty:
        selected_df.insert(0, "recommendation_rank", range(1, len(selected_df) + 1))
    return selected_df, verified_count, rejected_count


def _plot_recommended_tradeoff(candidates_df, recommended_df):
    """Plot predicted candidate performance and highlight the recommendations."""
    feasible_df = candidates_df[candidates_df["feasible"]]
    infeasible_df = candidates_df[~candidates_df["feasible"]]

    plt.figure(figsize=(7, 5))
    plt.scatter(
        infeasible_df["predicted_pressure_drop_kPa"],
        infeasible_df["predicted_max_temperature_C"],
        s=8,
        alpha=0.18,
        color="tab:gray",
        label="Predicted infeasible",
    )
    plt.scatter(
        feasible_df["predicted_pressure_drop_kPa"],
        feasible_df["predicted_max_temperature_C"],
        s=8,
        alpha=0.22,
        color="tab:green",
        label="Predicted feasible",
    )
    plt.scatter(
        recommended_df["predicted_pressure_drop_kPa"],
        recommended_df["predicted_max_temperature_C"],
        s=42,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.2,
        label="Recommended top 20",
    )
    plt.xlabel("Predicted pressure drop / kPa")
    plt.ylabel("Predicted maximum temperature / degC")
    plt.title("AI-recommended cold-plate designs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "recommended_designs_tradeoff.png", dpi=300)
    plt.close()


def _plot_surrogate_vs_physics(recommended_df):
    """Create a diagnostic comparison of screening predictions and verification."""
    recommendation_ids = np.arange(1, len(recommended_df) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(
        recommendation_ids,
        recommended_df["predicted_max_temperature_C"],
        marker="o",
        label="AI prediction",
    )
    axes[0].plot(
        recommendation_ids,
        recommended_df["physics_max_temperature_C"],
        marker="s",
        label="Physics verification",
    )
    axes[0].set_xlabel("Recommendation rank")
    axes[0].set_ylabel("Maximum temperature / degC")
    axes[0].set_title("Temperature screening diagnostic")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(
        recommendation_ids,
        recommended_df["predicted_pressure_drop_kPa"],
        marker="o",
        s=36,
        label="AI prediction",
    )
    axes[1].scatter(
        recommendation_ids,
        recommended_df["physics_pressure_drop_kPa"],
        marker="s",
        s=36,
        label="Physics verification",
    )
    axes[1].set_xlabel("Recommendation rank")
    axes[1].set_ylabel("Pressure drop / kPa")
    axes[1].set_title("Pressure-drop screening diagnostic")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        "Surrogate screening diagnostic\n"
        "AI prediction is used only for screening; final recommendation is ranked by physics verification.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "surrogate_vs_physics_verification.png", dpi=300)
    plt.close(fig)


def _plot_engineering_tradeoff(recommended_df):
    """Plot verified thermal-hydraulic performance with flow-rate-sized points."""
    point_sizes = 35 + 25 * recommended_df["flow_rate_L_min"]

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        recommended_df["physics_pressure_drop_kPa"],
        recommended_df["physics_max_temperature_C"],
        s=point_sizes,
        c=recommended_df["flow_rate_L_min"],
        cmap="viridis",
        alpha=0.78,
        edgecolors="black",
        linewidths=0.4,
    )
    plt.colorbar(scatter, label="Flow rate / L min$^{-1}$")
    plt.xlabel("Physics pressure drop / kPa")
    plt.ylabel("Physics maximum temperature / degC")
    for _, row in recommended_df.head(5).iterrows():
        plt.annotate(
            str(int(row["recommendation_rank"])),
            (
                row["physics_pressure_drop_kPa"],
                row["physics_max_temperature_C"],
            ),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
            weight="bold",
        )
    plt.title("Final recommended designs based on physics verification")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "recommended_designs_engineering_tradeoff.png", dpi=300)
    plt.savefig(
        FIGURE_DIR / "final_recommended_designs_physics_tradeoff.png",
        dpi=300,
    )
    plt.close()


def _plot_parameter_distribution(recommended_df):
    """Plot distributions of the four parameters used by the recommender."""
    plots = [
        ("flow_rate_L_min", "Flow rate / L min$^{-1}$"),
        ("channel_width_mm", "Channel width / mm"),
        ("channel_height_mm", "Channel height / mm"),
        ("channel_number", "Channel number"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))

    for axis, (column, label) in zip(axes.flat, plots):
        axis.hist(recommended_df[column], bins=8, color="tab:blue", alpha=0.78)
        axis.set_xlabel(label)
        axis.set_ylabel("Recommended design count")
        axis.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Recommended design parameter distribution")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "recommended_designs_parameter_distribution.png",
        dpi=300,
    )
    plt.close(fig)


def _save_v1_v2_comparison(v1_verified_df, v2_verified_df):
    """Save verified mean performance for the original and engineering scores."""
    records = []
    for recommender_version, df in [
        ("v1_temperature_pressure_score", v1_verified_df),
        ("v2_engineering_score", v2_verified_df),
    ]:
        records.append(
            {
                "recommender_version": recommender_version,
                "recommended_design_count": len(df),
                "mean_flow_rate_L_min": df["flow_rate_L_min"].mean(),
                "mean_physics_max_temperature_C": df[
                    "physics_max_temperature_C"
                ].mean(),
                "mean_physics_pressure_drop_kPa": df[
                    "physics_pressure_drop_kPa"
                ].mean(),
            }
        )

    comparison_df = pd.DataFrame(records)
    comparison_df.to_csv(DATA_DIR / "recommended_designs_v1_vs_v2.csv", index=False)
    return comparison_df


def _print_top_designs(recommended_df):
    """Print a concise view of the first five recommendations."""
    print("\nTop 5 recommended designs:")
    for rank, (_, design) in enumerate(recommended_df.head(5).iterrows(), start=1):
        print(
            f"{rank}. flow={design['flow_rate_L_min']:.3f} L/min, "
            f"channels={int(design['channel_number'])}, "
            f"width={design['channel_width_mm']:.3f} mm, "
            f"height={design['channel_height_mm']:.3f} mm, "
            f"predicted Tmax={design['predicted_max_temperature_C']:.3f} degC, "
            f"predicted pressure drop={design['predicted_pressure_drop_kPa']:.4f} kPa, "
            f"physics engineering score={design['physics_engineering_score']:.6f}"
        )


def main():
    """Generate, rank, and verify AI-assisted cold-plate recommendations."""
    _prepare_output_dirs()
    baseline_case = load_case(CASE_PATH)
    temperature_bundle = _load_model_bundle(
        TEMPERATURE_MODEL_PATH,
        "max_temperature_C",
    )
    pressure_bundle = _load_model_bundle(
        PRESSURE_MODEL_PATH,
        "pressure_drop_kPa",
    )
    candidates_df, generated_count = _generate_candidates(baseline_case)

    candidates_df["predicted_max_temperature_C"] = _predict(
        temperature_bundle,
        candidates_df,
    )
    candidates_df["predicted_pressure_drop_kPa_raw"] = _predict_restored_raw(
        pressure_bundle,
        candidates_df,
    )
    candidates_df["predicted_pressure_drop_kPa"] = np.maximum(
        candidates_df["predicted_pressure_drop_kPa_raw"],
        PRESSURE_FLOOR_KPA,
    )
    targets = baseline_case["targets"]
    candidates_df["predicted_thermal_margin_C"] = (
        targets["max_temperature_C"]
        - candidates_df["predicted_max_temperature_C"]
    )
    candidates_df["predicted_pressure_margin_kPa"] = (
        targets["max_pressure_drop_kPa"]
        - candidates_df["predicted_pressure_drop_kPa"]
    )
    candidates_df["thermal_ok"] = (
        candidates_df["predicted_max_temperature_C"]
        <= targets["max_temperature_C"] - SURROGATE_TEMPERATURE_SAFETY_MARGIN_C
    )
    candidates_df["pressure_ok"] = (
        candidates_df["predicted_pressure_drop_kPa"]
        <= targets["max_pressure_drop_kPa"] - SURROGATE_PRESSURE_SAFETY_MARGIN_KPA
    )
    candidates_df["feasible"] = (
        candidates_df["thermal_ok"] & candidates_df["pressure_ok"]
    )
    candidates_df.to_csv(
        DATA_DIR / "recommender_candidate_predictions.csv",
        index=False,
    )

    feasible_df = candidates_df[candidates_df["feasible"]]
    if feasible_df.empty:
        raise ValueError("No surrogate-predicted feasible designs were found.")

    v1_scored_df = _score_feasible_designs(feasible_df)
    v1_recommended_df = v1_scored_df.nsmallest(RECOMMENDATION_COUNT, "v1_score").copy()
    v1_verified_df = _verify_with_physics_model(v1_recommended_df, baseline_case)

    scored_df = _score_engineering_designs(
        feasible_df,
        baseline_case["chip"]["width_mm"],
        targets["max_temperature_C"],
    )
    ai_prefilter_df = scored_df.nsmallest(AI_PREFILTER_COUNT, "engineering_score")
    verified_prefilter_df = _verify_with_physics_model(
        ai_prefilter_df,
        baseline_case,
    )
    physics_verified_count = len(verified_prefilter_df)
    physics_feasible_df = verified_prefilter_df[
        verified_prefilter_df["physics_feasible"]
    ].copy()
    physics_rejected_count = physics_verified_count - len(physics_feasible_df)
    if physics_feasible_df.empty:
        raise ValueError("No physics-feasible recommendations were found.")

    verified_df = _score_physics_engineering_designs(
        physics_feasible_df,
        targets["max_temperature_C"],
    ).nsmallest(RECOMMENDATION_COUNT, "physics_engineering_score")
    verified_df = verified_df.copy()
    verified_df.insert(0, "recommendation_rank", range(1, len(verified_df) + 1))
    output_columns = [
        "recommendation_rank",
        "flow_rate_L_min",
        "channel_number",
        "channel_width_mm",
        "channel_height_mm",
        "required_width_mm",
        "predicted_max_temperature_C",
        "predicted_pressure_drop_kPa_raw",
        "predicted_pressure_drop_kPa",
        "predicted_thermal_margin_C",
        "predicted_pressure_margin_kPa",
        "manufacturing_score",
        "engineering_score",
        "physics_engineering_score",
        "physics_max_temperature_C",
        "physics_pressure_drop_kPa",
        "physics_pumping_power_W",
        "physics_Re",
        "physics_flow_regime",
        "physics_thermal_ok",
        "physics_pressure_ok",
        "physics_feasible",
    ]
    verified_df[output_columns].to_csv(DATA_DIR / "recommended_designs.csv", index=False)
    comparison_df = _save_v1_v2_comparison(v1_verified_df, verified_df)

    _plot_recommended_tradeoff(candidates_df, verified_df)
    _plot_surrogate_vs_physics(verified_df)
    _plot_engineering_tradeoff(verified_df)
    _plot_parameter_distribution(verified_df)

    temperature_mae = np.mean(
        np.abs(
            verified_df["predicted_max_temperature_C"]
            - verified_df["physics_max_temperature_C"]
        )
    )
    pressure_mae = np.mean(
        np.abs(
            verified_df["predicted_pressure_drop_kPa"]
            - verified_df["physics_pressure_drop_kPa"]
        )
    )
    print("\n========== AI Design Recommender Summary ==========")
    print(f"Candidate designs generated: {generated_count}")
    print(f"Designs passing geometric constraint: {len(candidates_df)}")
    print(f"AI-screened feasible designs with safety margins: {len(feasible_df)}")
    print(f"AI-prefiltered designs sent to physics verification: {physics_verified_count}")
    print(f"Physics-feasible designs after verification: {len(physics_feasible_df)}")
    print(f"Designs rejected by physics verification: {physics_rejected_count}")
    print(f"Final physics-feasible recommendations: {len(verified_df)}")
    _print_top_designs(verified_df)
    print("\nMean absolute surrogate-vs-physics error for recommended designs:")
    print(f"Temperature MAE: {temperature_mae:.6f} degC")
    print(f"Pressure-drop MAE: {pressure_mae:.6f} kPa")
    print("\nVerified v1 vs v2 recommendation comparison:")
    print(comparison_df.to_string(index=False))
    print("\nGenerated files:")
    print("1. data/recommender_candidate_predictions.csv")
    print("2. data/recommended_designs.csv")
    print("3. figures/recommended_designs_tradeoff.png")
    print("4. figures/surrogate_vs_physics_verification.png")
    print("5. data/recommended_designs_v1_vs_v2.csv")
    print("6. figures/recommended_designs_engineering_tradeoff.png")
    print("7. figures/final_recommended_designs_physics_tradeoff.png")
    print("8. figures/recommended_designs_parameter_distribution.png")


if __name__ == "__main__":
    main()
