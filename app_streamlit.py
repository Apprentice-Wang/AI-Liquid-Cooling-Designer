from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import load_case
from src.design_recommender import (
    PRESSURE_MODEL_PATH,
    TEMPERATURE_MODEL_PATH,
    _load_model_bundle,
    _plot_engineering_tradeoff,
    _plot_recommended_tradeoff,
    _plot_surrogate_vs_physics,
    _predict,
    _required_width_mm,
    _score_engineering_designs,
    _select_physics_verified_recommendations,
)


CASE_PATH = PROJECT_ROOT / "examples" / "single_chip_case.yaml"
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figures"
RECOMMENDATION_COUNT = 20
CHANNEL_SPACING_MM = 0.5
RANDOM_SEED = 42


def build_sidebar_inputs():
    """Collect user-adjustable design and recommender settings."""
    st.sidebar.header("Input Parameters")

    inputs = {
        "chip_power_W": st.sidebar.number_input(
            "chip_power_W",
            min_value=1.0,
            value=500.0,
            step=10.0,
        ),
        "target_max_temperature_C": st.sidebar.number_input(
            "target_max_temperature_C",
            min_value=0.0,
            value=85.0,
            step=1.0,
        ),
        "target_max_pressure_drop_kPa": st.sidebar.number_input(
            "target_max_pressure_drop_kPa",
            min_value=0.0,
            value=50.0,
            step=1.0,
        ),
        "flow_rate_min_L_min": st.sidebar.number_input(
            "flow_rate_min_L_min",
            min_value=0.01,
            value=0.5,
            step=0.1,
        ),
        "flow_rate_max_L_min": st.sidebar.number_input(
            "flow_rate_max_L_min",
            min_value=0.01,
            value=5.0,
            step=0.1,
        ),
        "channel_width_min_mm": st.sidebar.number_input(
            "channel_width_min_mm",
            min_value=0.01,
            value=0.5,
            step=0.1,
        ),
        "channel_width_max_mm": st.sidebar.number_input(
            "channel_width_max_mm",
            min_value=0.01,
            value=3.0,
            step=0.1,
        ),
        "channel_height_min_mm": st.sidebar.number_input(
            "channel_height_min_mm",
            min_value=0.01,
            value=0.5,
            step=0.1,
        ),
        "channel_height_max_mm": st.sidebar.number_input(
            "channel_height_max_mm",
            min_value=0.01,
            value=4.0,
            step=0.1,
        ),
        "candidate_number": st.sidebar.number_input(
            "candidate_number",
            min_value=1000,
            value=50000,
            step=1000,
        ),
        "temperature_safety_margin_C": st.sidebar.number_input(
            "temperature_safety_margin_C",
            min_value=0.0,
            value=3.0,
            step=0.5,
        ),
        "pressure_safety_margin_kPa": st.sidebar.number_input(
            "pressure_safety_margin_kPa",
            min_value=0.0,
            value=1.0,
            step=0.5,
        ),
    }
    return inputs


def validate_inputs(inputs):
    """Return a list of user-facing validation errors for invalid ranges."""
    errors = []
    if inputs["flow_rate_min_L_min"] >= inputs["flow_rate_max_L_min"]:
        errors.append("flow_rate_min_L_min must be smaller than flow_rate_max_L_min.")
    if inputs["channel_width_min_mm"] >= inputs["channel_width_max_mm"]:
        errors.append("channel_width_min_mm must be smaller than channel_width_max_mm.")
    if inputs["channel_height_min_mm"] >= inputs["channel_height_max_mm"]:
        errors.append(
            "channel_height_min_mm must be smaller than channel_height_max_mm."
        )
    if (
        inputs["target_max_temperature_C"]
        <= inputs["temperature_safety_margin_C"]
    ):
        errors.append(
            "temperature_safety_margin_C must be smaller than target_max_temperature_C."
        )
    if (
        inputs["target_max_pressure_drop_kPa"]
        <= inputs["pressure_safety_margin_kPa"]
    ):
        errors.append(
            "pressure_safety_margin_kPa must be smaller than target_max_pressure_drop_kPa."
        )
    return errors


def build_temporary_case(inputs):
    """Create an in-memory baseline case without editing the YAML file."""
    baseline_case = deepcopy(load_case(CASE_PATH))
    baseline_case["chip"]["power_W"] = float(inputs["chip_power_W"])
    baseline_case["targets"]["max_temperature_C"] = float(
        inputs["target_max_temperature_C"]
    )
    baseline_case["targets"]["max_pressure_drop_kPa"] = float(
        inputs["target_max_pressure_drop_kPa"]
    )
    return baseline_case


def generate_candidates(baseline_case, inputs):
    """Generate random candidates from the UI ranges and apply geometry limits."""
    rng = np.random.default_rng(RANDOM_SEED)
    candidate_count = int(inputs["candidate_number"])
    candidates = pd.DataFrame(
        {
            "flow_rate_L_min": rng.uniform(
                inputs["flow_rate_min_L_min"],
                inputs["flow_rate_max_L_min"],
                candidate_count,
            ),
            "channel_number": rng.integers(4, 31, candidate_count),
            "channel_width_mm": rng.uniform(
                inputs["channel_width_min_mm"],
                inputs["channel_width_max_mm"],
                candidate_count,
            ),
            "channel_height_mm": rng.uniform(
                inputs["channel_height_min_mm"],
                inputs["channel_height_max_mm"],
                candidate_count,
            ),
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


@st.cache_resource(show_spinner=False)
def load_model_bundles():
    """Load surrogate model bundles once per Streamlit session."""
    temperature_bundle = _load_model_bundle(
        TEMPERATURE_MODEL_PATH,
        "max_temperature_C",
    )
    pressure_bundle = _load_model_bundle(
        PRESSURE_MODEL_PATH,
        "pressure_drop_kPa",
    )
    return temperature_bundle, pressure_bundle


def run_recommendation(inputs):
    """Run the engineering-oriented recommender with UI-provided settings."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    baseline_case = build_temporary_case(inputs)
    candidates_df, generated_count = generate_candidates(baseline_case, inputs)
    if candidates_df.empty:
        raise ValueError("No candidates passed the geometric width constraint.")

    temperature_bundle, pressure_bundle = load_model_bundles()
    candidates_df["predicted_max_temperature_C"] = _predict(
        temperature_bundle,
        candidates_df,
    )
    candidates_df["predicted_pressure_drop_kPa"] = _predict(
        pressure_bundle,
        candidates_df,
    )

    targets = baseline_case["targets"]
    candidates_df["predicted_thermal_margin_C"] = (
        targets["max_temperature_C"] - candidates_df["predicted_max_temperature_C"]
    )
    candidates_df["predicted_pressure_margin_kPa"] = (
        targets["max_pressure_drop_kPa"] - candidates_df["predicted_pressure_drop_kPa"]
    )
    candidates_df["thermal_ok"] = (
        candidates_df["predicted_max_temperature_C"]
        <= targets["max_temperature_C"] - inputs["temperature_safety_margin_C"]
    )
    candidates_df["pressure_ok"] = (
        candidates_df["predicted_pressure_drop_kPa"]
        <= targets["max_pressure_drop_kPa"] - inputs["pressure_safety_margin_kPa"]
    )
    candidates_df["feasible"] = candidates_df["thermal_ok"] & candidates_df["pressure_ok"]
    candidates_df.to_csv(
        DATA_DIR / "recommender_candidate_predictions.csv",
        index=False,
    )

    feasible_df = candidates_df[candidates_df["feasible"]]
    if feasible_df.empty:
        raise ValueError("No AI-screened feasible designs were found.")

    scored_df = _score_engineering_designs(
        feasible_df,
        baseline_case["chip"]["width_mm"],
        targets["max_temperature_C"],
    )
    ranked_candidates_df = scored_df.sort_values("engineering_score")
    verified_df, physics_checked_count, physics_rejected_count = (
        _select_physics_verified_recommendations(
            ranked_candidates_df,
            baseline_case,
            RECOMMENDATION_COUNT,
        )
    )
    if verified_df.empty:
        raise ValueError("No physics-feasible recommendations were found.")

    output_columns = [
        "recommendation_rank",
        "flow_rate_L_min",
        "channel_number",
        "channel_width_mm",
        "channel_height_mm",
        "required_width_mm",
        "predicted_max_temperature_C",
        "predicted_pressure_drop_kPa",
        "predicted_thermal_margin_C",
        "predicted_pressure_margin_kPa",
        "manufacturing_score",
        "engineering_score",
        "physics_max_temperature_C",
        "physics_pressure_drop_kPa",
        "physics_pumping_power_W",
        "physics_Re",
        "physics_flow_regime",
        "physics_thermal_ok",
        "physics_pressure_ok",
        "physics_feasible",
    ]
    verified_df = verified_df[output_columns].copy()
    verified_df.to_csv(DATA_DIR / "recommended_designs.csv", index=False)

    _plot_recommended_tradeoff(candidates_df, verified_df)
    _plot_surrogate_vs_physics(verified_df)
    _plot_engineering_tradeoff(verified_df)

    run_info = {
        "generated_count": generated_count,
        "geometry_valid_count": len(candidates_df),
        "ai_feasible_count": len(feasible_df),
        "physics_checked_count": physics_checked_count,
        "physics_rejected_count": physics_rejected_count,
    }
    return verified_df, run_info


def display_metrics(recommended_df):
    """Show high-level recommendation metrics."""
    metric_columns = st.columns(3)
    metric_columns[0].metric("Recommended designs", len(recommended_df))
    metric_columns[1].metric(
        "Lowest physics Tmax",
        f"{recommended_df['physics_max_temperature_C'].min():.2f} °C",
    )
    metric_columns[2].metric(
        "Highest physics Tmax",
        f"{recommended_df['physics_max_temperature_C'].max():.2f} °C",
    )

    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Average physics Tmax",
        f"{recommended_df['physics_max_temperature_C'].mean():.2f} °C",
    )
    metric_columns[1].metric(
        "Average pressure drop",
        f"{recommended_df['physics_pressure_drop_kPa'].mean():.3f} kPa",
    )
    metric_columns[2].metric(
        "Average flow rate",
        f"{recommended_df['flow_rate_L_min'].mean():.3f} L/min",
    )


def display_best_design(recommended_df):
    """Show the rank-1 recommended design in a compact table."""
    best_design = recommended_df.sort_values("recommendation_rank").iloc[0]
    best_columns = [
        "flow_rate_L_min",
        "channel_number",
        "channel_width_mm",
        "channel_height_mm",
        "physics_max_temperature_C",
        "physics_pressure_drop_kPa",
        "physics_pumping_power_W",
        "physics_Re",
        "physics_flow_regime",
    ]
    st.subheader("Best Recommended Design")
    st.dataframe(
        best_design[best_columns].to_frame("value"),
        use_container_width=True,
    )


def display_image(path, caption):
    """Display a generated figure when it exists, otherwise show a gentle hint."""
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{path.name} does not exist yet. Run AI Recommendation first.")


def display_limitations():
    """Display current model limitations at the bottom of the page."""
    st.markdown(
        """
### Model Limitations

- This is an early-stage design screening tool.
- The current results are based on simplified thermal-hydraulic models and surrogate models.
- Final designs should be validated by CFD simulation or experiments.
- Manifold effects, contact thermal resistance, non-uniform heat sources, and temperature-dependent fluid properties are not fully considered in the current version.
"""
    )


def main():
    """Render the Streamlit interface."""
    st.set_page_config(
        page_title="AI-Liquid-Cooling-Designer",
        layout="wide",
    )
    st.title("AI-Liquid-Cooling-Designer")
    st.write(
        "This tool provides early-stage liquid cold plate design screening using "
        "physics-based calculation, machine learning surrogate models, and "
        "engineering-oriented design recommendation."
    )

    inputs = build_sidebar_inputs()

    st.subheader("Project Workflow")
    st.markdown(
        "**Physics model → Design space search → Surrogate model → "
        "AI recommender → Physics verification**"
    )

    run_button = st.button("Run AI Recommendation", type="primary")
    if run_button:
        errors = validate_inputs(inputs)
        if errors:
            for error in errors:
                st.error(error)
        else:
            with st.spinner("Running AI recommendation and physics verification..."):
                try:
                    recommended_df, run_info = run_recommendation(inputs)
                except Exception as exc:
                    st.error(f"Recommendation failed: {exc}")
                else:
                    st.success("AI recommendation completed.")
                    st.caption(
                        "Generated candidates: "
                        f"{run_info['generated_count']} | "
                        "Geometry-valid candidates: "
                        f"{run_info['geometry_valid_count']} | "
                        "AI-screened feasible candidates: "
                        f"{run_info['ai_feasible_count']} | "
                        "Physics checked: "
                        f"{run_info['physics_checked_count']} | "
                        "Physics rejected: "
                        f"{run_info['physics_rejected_count']}"
                    )
                    st.session_state["recommended_df"] = recommended_df

    recommended_df = st.session_state.get("recommended_df")
    existing_csv = DATA_DIR / "recommended_designs.csv"
    if recommended_df is None and existing_csv.exists():
        recommended_df = pd.read_csv(existing_csv)

    if recommended_df is not None and not recommended_df.empty:
        st.subheader("Recommended Designs")
        display_metrics(recommended_df)
        display_best_design(recommended_df)
        st.dataframe(recommended_df, use_container_width=True)

        csv_bytes = recommended_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download recommended_designs.csv",
            data=csv_bytes,
            file_name="recommended_designs.csv",
            mime="text/csv",
        )

    st.subheader("Generated Figures")
    figure_columns = st.columns(2)
    with figure_columns[0]:
        display_image(
            FIGURE_DIR / "final_recommended_designs_physics_tradeoff.png",
            "Final recommended designs based on physics verification",
        )
    with figure_columns[1]:
        display_image(
            FIGURE_DIR / "surrogate_vs_physics_verification.png",
            "Surrogate screening diagnostic",
        )

    display_limitations()


if __name__ == "__main__":
    main()
