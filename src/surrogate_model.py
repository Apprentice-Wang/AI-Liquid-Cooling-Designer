from pathlib import Path

import joblib
import matplotlib

# Use a non-interactive backend so batch training can save figures reliably.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "design_space_results.csv"
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = [
    "flow_rate_L_min",
    "channel_number",
    "channel_width_mm",
    "channel_height_mm",
]

TARGETS = {
    "temperature": {
        "column": "max_temperature_C",
        "label": "Maximum temperature / degC",
        "prediction_figure": "surrogate_temperature_prediction.png",
        "importance_figure": "feature_importance_temperature.png",
        "model_file": "temperature_model.pkl",
    },
    "pressure_drop": {
        "column": "pressure_drop_kPa",
        "label": "Pressure drop / kPa",
        "importance_figure": "feature_importance_pressure.png",
        "model_file": "pressure_drop_model.pkl",
    },
}

MIN_LOW_PRESSURE_SAMPLE_COUNT = 20


def _prepare_output_dirs():
    """Create folders used for metrics, diagnostic figures, and trained models."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _load_design_space():
    """Load and validate the columns required for surrogate-model training."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cannot find data/design_space_results.csv. "
            "Run python src/design_space_search.py first."
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = FEATURE_COLUMNS + [
        target_config["column"] for target_config in TARGETS.values()
    ]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns in design-space data: "
            + ", ".join(missing_columns)
        )

    # Drop incomplete records so every candidate model sees identical samples.
    model_df = df[required_columns].dropna()
    if model_df.empty:
        raise ValueError("No complete design-space records are available for training.")

    return model_df


def _candidate_models():
    """Build a fresh pair of reproducible regression models for one target."""
    return {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=42,
        ),
    }


def _pressure_candidate_models():
    """
    Build raw-scale and log1p pressure-drop models.

    TransformedTargetRegressor applies log1p during fitting and expm1 during
    prediction, so saved log models still return pressure drop directly in kPa.
    """
    models = {}
    for base_name, model in _candidate_models().items():
        models[f"{base_name}_raw"] = {
            "model": model,
            "transform": "raw",
        }

    for base_name, model in _candidate_models().items():
        models[f"{base_name}_log1p"] = {
            "model": TransformedTargetRegressor(
                regressor=model,
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "transform": "log1p",
        }

    return models


def _calculate_metrics(y_true, y_pred):
    """Return the standard regression metrics requested for model comparison."""
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def _metric_record(target, model, transform, evaluation_subset, y_true, y_pred):
    """Create one metrics row, including the evaluated subset sample count."""
    metrics = _calculate_metrics(y_true, y_pred)
    return {
        "target": target,
        "model": model,
        "transform": transform,
        "evaluation_subset": evaluation_subset,
        "sample_count": len(y_true),
        **metrics,
    }


def _plot_prediction(y_true, y_pred, label, model_name, output_path):
    """Plot measured model outputs against surrogate-model predictions."""
    lower_bound = min(y_true.min(), y_pred.min())
    upper_bound = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, s=20, alpha=0.55, color="tab:blue")
    plt.plot(
        [lower_bound, upper_bound],
        [lower_bound, upper_bound],
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="Ideal prediction",
    )
    plt.xlabel(f"True {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(f"Surrogate prediction: {model_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_pressure_zoom(y_true, y_pred, model_name, output_path):
    """Plot pressure predictions only in the engineering-relevant 0-100 kPa range."""
    mask = (y_true >= 0) & (y_true <= 100)
    zoom_true = y_true[mask]
    zoom_pred = y_pred[mask]

    plt.figure(figsize=(6, 5))
    plt.scatter(zoom_true, zoom_pred, s=20, alpha=0.55, color="tab:blue")
    plt.plot([0, 100], [0, 100], color="tab:red", linestyle="--", linewidth=1.5)
    plt.xlim(0, 100)
    plt.xlabel("True pressure drop / kPa")
    plt.ylabel("Predicted pressure drop / kPa")
    plt.title(f"Pressure prediction in 0-100 kPa range: {model_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_pressure_residuals(y_true, y_pred, model_name, output_path):
    """Plot pressure residuals against true pressure to reveal biased regions."""
    residuals = y_pred - y_true

    plt.figure(figsize=(6.5, 5))
    plt.scatter(y_true, residuals, s=20, alpha=0.55, color="tab:purple")
    plt.axhline(0, color="tab:red", linestyle="--", linewidth=1.5)
    plt.xlabel("True pressure drop / kPa")
    plt.ylabel("Residual: predicted - true / kPa")
    plt.title(f"Pressure residuals: {model_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _feature_importances(model):
    """Return importances from a raw model or a fitted transformed-target wrapper."""
    fitted_model = getattr(model, "regressor_", model)
    return fitted_model.feature_importances_


def _model_bundle(model, target, transform):
    """
    Package a fitted model with the metadata required by downstream tools.

    A fitted log1p pressure model is stored as its inner regressor because
    design_recommender.py performs the inverse expm1 transform explicitly.
    """
    bundled_model = getattr(model, "regressor_", model) if transform == "log1p" else model
    return {
        "model": bundled_model,
        "features": FEATURE_COLUMNS,
        "target": target,
        "transform": transform,
    }


def _plot_feature_importance(model, target_name, output_path):
    """Plot feature importance from the selected tree-based surrogate model."""
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": _feature_importances(model),
        }
    ).sort_values("importance")

    plt.figure(figsize=(7, 4.5))
    plt.barh(
        importance_df["feature"],
        importance_df["importance"],
        color="tab:green",
    )
    plt.xlabel("Feature importance")
    plt.title(f"Feature importance: {target_name}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_surrogate_models():
    """
    Train thermal and hydraulic surrogate models and save diagnostics.

    Temperature retains the original raw-target comparison. Pressure drop adds
    log1p candidates and low-pressure metrics to handle its skewed distribution.
    """
    _prepare_output_dirs()
    df = _load_design_space()
    x = df[FEATURE_COLUMNS]
    train_indices, test_indices = train_test_split(
        df.index,
        test_size=0.2,
        random_state=42,
    )
    x_train = x.loc[train_indices]
    x_test = x.loc[test_indices]

    metric_records = []

    # Preserve the original temperature-model training and R2-based selection.
    temperature_config = TARGETS["temperature"]
    temperature_column = temperature_config["column"]
    temperature_y_train = df.loc[train_indices, temperature_column]
    temperature_y_test = df.loc[test_indices, temperature_column]
    temperature_results = []

    for model_name, model in _candidate_models().items():
        model.fit(x_train, temperature_y_train)
        y_pred = model.predict(x_test)
        metrics = _metric_record(
            temperature_column,
            model_name,
            "raw",
            "overall",
            temperature_y_test,
            y_pred,
        )
        metric_records.append(metrics)
        temperature_results.append((metrics["R2"], model_name, model, y_pred, metrics))
        _print_metrics(metrics)

    (
        _,
        selected_temperature_name,
        selected_temperature_model,
        selected_temperature_prediction,
        selected_temperature_metrics,
    ) = max(temperature_results, key=lambda result: result[0])
    joblib.dump(
        _model_bundle(selected_temperature_model, temperature_column, "none"),
        MODEL_DIR / temperature_config["model_file"],
    )
    _plot_prediction(
        temperature_y_test,
        selected_temperature_prediction,
        temperature_config["label"],
        selected_temperature_name,
        FIGURE_DIR / temperature_config["prediction_figure"],
    )
    _plot_feature_importance(
        selected_temperature_model,
        temperature_column,
        FIGURE_DIR / temperature_config["importance_figure"],
    )

    # Compare raw and log1p pressure models on overall and low-pressure subsets.
    pressure_config = TARGETS["pressure_drop"]
    pressure_column = pressure_config["column"]
    pressure_y_train = df.loc[train_indices, pressure_column]
    pressure_y_test = df.loc[test_indices, pressure_column]
    pressure_results = []

    for model_name, candidate in _pressure_candidate_models().items():
        model = candidate["model"]
        transform = candidate["transform"]
        model.fit(x_train, pressure_y_train)
        y_pred = model.predict(x_test)
        predictions = pd.Series(y_pred, index=pressure_y_test.index)
        subset_metrics = {}

        for subset_name, mask in [
            ("overall", pd.Series(True, index=pressure_y_test.index)),
            ("pressure_drop_le_50_kPa", pressure_y_test <= 50),
            ("pressure_drop_le_100_kPa", pressure_y_test <= 100),
        ]:
            metrics = _metric_record(
                pressure_column,
                model_name,
                transform,
                subset_name,
                pressure_y_test[mask],
                predictions[mask],
            )
            metric_records.append(metrics)
            subset_metrics[subset_name] = metrics
            _print_metrics(metrics)

        pressure_results.append(
            {
                "name": model_name,
                "transform": transform,
                "model": model,
                "prediction": predictions,
                "metrics": subset_metrics,
            }
        )

    best_pressure_overall = min(
        pressure_results,
        key=lambda result: result["metrics"]["overall"]["MAE"],
    )
    best_pressure_100 = min(
        pressure_results,
        key=lambda result: result["metrics"]["pressure_drop_le_100_kPa"]["MAE"],
    )
    low_pressure_count = best_pressure_100["metrics"]["pressure_drop_le_100_kPa"][
        "sample_count"
    ]

    if low_pressure_count >= MIN_LOW_PRESSURE_SAMPLE_COUNT:
        selected_pressure = best_pressure_100
        pressure_selection_reason = (
            "Selected the model with the lowest MAE on the true pressure-drop "
            f"<= 100 kPa test subset ({low_pressure_count} samples)."
        )
    else:
        selected_pressure = best_pressure_overall
        pressure_selection_reason = (
            "The true pressure-drop <= 100 kPa test subset contained only "
            f"{low_pressure_count} samples, below the minimum of "
            f"{MIN_LOW_PRESSURE_SAMPLE_COUNT}; selected the lowest overall-MAE model."
        )

    joblib.dump(
        _model_bundle(
            selected_pressure["model"],
            pressure_column,
            selected_pressure["transform"],
        ),
        MODEL_DIR / pressure_config["model_file"],
    )
    _save_pressure_figures(
        pressure_results,
        selected_pressure,
        pressure_y_test,
        pressure_config,
    )

    selection_summary = _save_model_selection_summary(
        selected_temperature_name,
        selected_temperature_metrics,
        best_pressure_overall,
        best_pressure_100,
        selected_pressure,
        pressure_selection_reason,
    )

    metrics_df = pd.DataFrame(metric_records)
    metrics_df.to_csv(DATA_DIR / "surrogate_model_metrics.csv", index=False)
    return metrics_df, selection_summary


def _print_metrics(metrics):
    """Print one model evaluation row in a compact terminal-friendly format."""
    print(
        f"{metrics['target']} | {metrics['model']} | {metrics['transform']} | "
        f"{metrics['evaluation_subset']}: "
        f"n={metrics['sample_count']}, "
        f"R2={metrics['R2']:.6f}, "
        f"MAE={metrics['MAE']:.6f}, "
        f"RMSE={metrics['RMSE']:.6f}"
    )


def _save_pressure_figures(
    pressure_results,
    selected_pressure,
    pressure_y_test,
    pressure_config,
):
    """Save pressure diagnostics for the best raw, log1p, and final models."""
    best_raw = min(
        (result for result in pressure_results if result["transform"] == "raw"),
        key=lambda result: result["metrics"]["overall"]["MAE"],
    )
    best_log1p = min(
        (result for result in pressure_results if result["transform"] == "log1p"),
        key=lambda result: result["metrics"]["overall"]["MAE"],
    )

    _plot_prediction(
        pressure_y_test,
        best_raw["prediction"],
        pressure_config["label"],
        best_raw["name"],
        FIGURE_DIR / "surrogate_pressure_prediction_raw.png",
    )
    _plot_prediction(
        pressure_y_test,
        best_log1p["prediction"],
        pressure_config["label"],
        best_log1p["name"],
        FIGURE_DIR / "surrogate_pressure_prediction_log1p.png",
    )
    _plot_pressure_zoom(
        pressure_y_test,
        selected_pressure["prediction"],
        selected_pressure["name"],
        FIGURE_DIR / "surrogate_pressure_prediction_zoom_0_100kPa.png",
    )
    _plot_pressure_residuals(
        pressure_y_test,
        selected_pressure["prediction"],
        selected_pressure["name"],
        FIGURE_DIR / "pressure_residual_vs_true.png",
    )
    _plot_feature_importance(
        selected_pressure["model"],
        pressure_config["column"],
        FIGURE_DIR / pressure_config["importance_figure"],
    )


def _save_model_selection_summary(
    temperature_name,
    temperature_metrics,
    best_pressure_overall,
    best_pressure_100,
    selected_pressure,
    pressure_selection_reason,
):
    """Write a readable record of the model-selection decision."""
    summary = {
        "temperature_name": temperature_name,
        "temperature_metrics": temperature_metrics,
        "pressure_overall_name": best_pressure_overall["name"],
        "pressure_overall_metrics": best_pressure_overall["metrics"]["overall"],
        "pressure_100_name": best_pressure_100["name"],
        "pressure_100_metrics": best_pressure_100["metrics"][
            "pressure_drop_le_100_kPa"
        ],
        "selected_pressure_name": selected_pressure["name"],
        "pressure_selection_reason": pressure_selection_reason,
    }
    lines = [
        "Surrogate Model Selection Summary",
        "=================================",
        (
            f"Temperature model: {temperature_name} "
            f"(overall R2={temperature_metrics['R2']:.6f}, "
            f"MAE={temperature_metrics['MAE']:.6f}, "
            f"RMSE={temperature_metrics['RMSE']:.6f})"
        ),
        (
            f"Pressure-drop overall best MAE model: {best_pressure_overall['name']} "
            f"(MAE={best_pressure_overall['metrics']['overall']['MAE']:.6f})"
        ),
        (
            "Pressure-drop <= 100 kPa best MAE model: "
            f"{best_pressure_100['name']} "
            f"(n={best_pressure_100['metrics']['pressure_drop_le_100_kPa']['sample_count']}, "
            f"MAE={best_pressure_100['metrics']['pressure_drop_le_100_kPa']['MAE']:.6f})"
        ),
        f"Saved pressure-drop model: {selected_pressure['name']}",
        f"Selection reason: {pressure_selection_reason}",
        (
            "Note: saved model bundles include feature order and transform "
            "metadata. Downstream tools apply expm1 to log1p model predictions."
        ),
    ]
    (MODEL_DIR / "model_selection_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    """Train and save thermal and hydraulic surrogate models."""
    metrics_df, summary = train_surrogate_models()

    print("\n========== Selected Surrogate Models ==========")
    temperature_metrics = summary["temperature_metrics"]
    print(
        f"Temperature best model: {summary['temperature_name']} "
        f"(R2={temperature_metrics['R2']:.6f}, "
        f"MAE={temperature_metrics['MAE']:.6f}, "
        f"RMSE={temperature_metrics['RMSE']:.6f})"
    )
    print(f"Pressure-drop overall best model: {summary['pressure_overall_name']}")
    print(
        "Pressure-drop <= 100 kPa best model: "
        f"{summary['pressure_100_name']}"
    )
    print(f"Saved pressure-drop model: {summary['selected_pressure_name']}")
    print(f"Selection reason: {summary['pressure_selection_reason']}")
    print(f"Evaluated model configurations: {len(metrics_df)}")
    print("\nGenerated files:")
    print("1. data/surrogate_model_metrics.csv")
    print("2. models/temperature_model.pkl")
    print("3. models/pressure_drop_model.pkl")
    print("4. models/model_selection_summary.txt")
    print("5. figures/surrogate_temperature_prediction.png")
    print("6. figures/surrogate_pressure_prediction_raw.png")
    print("7. figures/surrogate_pressure_prediction_log1p.png")
    print("8. figures/surrogate_pressure_prediction_zoom_0_100kPa.png")
    print("9. figures/pressure_residual_vs_true.png")
    print("10. figures/feature_importance_temperature.png")
    print("11. figures/feature_importance_pressure.png")


if __name__ == "__main__":
    main()
