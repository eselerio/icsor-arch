"""Build the full-size component-level non-negativity violation heatmap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


MODEL_ORDER = [
    "xgboost_regressor",
    "lightgbm_regressor",
    "catboost_regressor",
    "adaboost_regressor",
    "random_forest_regressor",
    "extra_trees_regressor",
    "svr_regressor",
    "knn_regressor",
    "pls_regressor",
    "multitask_elastic_net_regressor",
    "multitask_lasso_regressor",
    "ann_deep_regressor",
    "tabnet_regressor",
]
MODEL_LABELS = {
    "xgboost_regressor": "XGBoost",
    "lightgbm_regressor": "LightGBM",
    "catboost_regressor": "CatBoost",
    "adaboost_regressor": "AdaBoost",
    "random_forest_regressor": "Random Forest",
    "extra_trees_regressor": "Extra Trees",
    "svr_regressor": "SVR",
    "knn_regressor": "k-NN",
    "pls_regressor": "PLS",
    "multitask_elastic_net_regressor": "Multi-task Elastic Net",
    "multitask_lasso_regressor": "Multi-task Lasso",
    "ann_deep_regressor": "MLP",
    "tabnet_regressor": "TabNet",
}
COMPONENTS = [
    "S_O",
    "S_F",
    "S_A",
    "S_NH4",
    "S_NO2",
    "S_NO3",
    "S_N2",
    "S_PO4",
    "S_I",
    "S_ALK",
    "X_I",
    "X_S",
    "X_H",
    "X_PAO",
    "X_PP",
    "X_PHA",
    "X_AOB",
    "X_NOB",
    "X_MeP",
    "X_MeOH",
]
COMPONENT_LABELS = {
    "S_O": r"$S_O$",
    "S_F": r"$S_F$",
    "S_A": r"$S_A$",
    "S_NH4": r"$S_{NH4}$",
    "S_NO2": r"$S_{NO2}$",
    "S_NO3": r"$S_{NO3}$",
    "S_N2": r"$S_{N2}$",
    "S_PO4": r"$S_{PO4}$",
    "S_I": r"$S_I$",
    "S_ALK": r"$S_{ALK}$",
    "X_I": r"$X_I$",
    "X_S": r"$X_S$",
    "X_H": r"$X_H$",
    "X_PAO": r"$X_{PAO}$",
    "X_PP": r"$X_{PP}$",
    "X_PHA": r"$X_{PHA}$",
    "X_AOB": r"$X_{AOB}$",
    "X_NOB": r"$X_{NOB}$",
    "X_MeP": r"$X_{MeP}$",
    "X_MeOH": r"$X_{MeOH}$",
}
SAMPLE_SIZE = 10_000
FOLDS = range(5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    run_root = project_root / "results" / "article_final_v2"
    output_root = project_root / "article" / "wip"
    output_root.mkdir(parents=True, exist_ok=True)

    completed = json.loads((run_root / "COMPLETED.json").read_text(encoding="utf-8"))
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(completed.get("article_eligible") is True, "The completed run is not article eligible.")
    require(completed.get("manifest_sha256") == sha256(manifest_path), "Manifest digest mismatch.")
    require(manifest.get("run_id") == "article_final_v2", "Unexpected source run.")

    tau = float(json.loads((run_root / "inputs" / "params.resolved.json").read_text(encoding="utf-8"))["projection"]["tau"])
    stored_physical = pd.read_csv(run_root / "metrics" / "id_physical_fold.csv")
    stored_physical = stored_physical.query(
        "sample_size == @SAMPLE_SIZE and prediction_type == 'raw'"
    ).set_index(["model_key", "outer_fold"])

    fold_rows: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        for fold in FOLDS:
            relative = f"predictions/id/{model_key}_n{SAMPLE_SIZE}_fold{fold}.parquet"
            prediction_path = run_root / relative
            inventory = manifest["artifact_inventory"].get(relative)
            require(inventory is not None, f"Prediction is absent from the manifest: {relative}")
            require(sha256(prediction_path) == inventory["sha256"], f"Prediction digest mismatch: {relative}")

            columns = [f"raw_{name}" for name in COMPONENTS] + [f"projected_{name}" for name in COMPONENTS]
            prediction = pd.read_parquet(prediction_path, columns=columns)
            raw = prediction[[f"raw_{name}" for name in COMPONENTS]].to_numpy(float)
            projected = prediction[[f"projected_{name}" for name in COMPONENTS]].to_numpy(float)
            raw_violations = raw < -tau
            projected_violations = projected < -tau
            require(not np.any(projected_violations), f"Projected violation found in {relative}")

            stored_rate = float(stored_physical.loc[(model_key, fold), "nonnegative_violation_rate"])
            calculated_rate = float(np.mean(np.any(raw_violations, axis=1)))
            require(np.isclose(calculated_rate, stored_rate, rtol=0.0, atol=1e-12), f"Stored row violation rate does not reproduce: {relative}")

            for column_index, component in enumerate(COMPONENTS):
                count = int(raw_violations[:, column_index].sum())
                fold_rows.append(
                    {
                        "model": MODEL_LABELS[model_key],
                        "model_key": model_key,
                        "component": component,
                        "sample_size": SAMPLE_SIZE,
                        "outer_fold": fold,
                        "validation_predictions": len(prediction),
                        "raw_negative_predictions": count,
                        "raw_nonnegativity_violation_percent": 100.0 * count / len(prediction),
                    }
                )

    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby(["model", "model_key", "component", "sample_size"], sort=False)
        .agg(
            folds=("outer_fold", "nunique"),
            raw_negative_predictions=("raw_negative_predictions", "sum"),
            validation_predictions=("validation_predictions", "sum"),
            raw_nonnegativity_violation_percent_mean=("raw_nonnegativity_violation_percent", "mean"),
            raw_nonnegativity_violation_percent_sample_sd=("raw_nonnegativity_violation_percent", "std"),
        )
        .reset_index()
    )
    require(len(summary) == len(MODEL_ORDER) * len(COMPONENTS), "Incomplete heatmap grid.")
    summary.to_csv(output_root / "source_component_nonnegativity_violations_id.csv", index=False)

    matrix = (
        summary.pivot(index="model_key", columns="component", values="raw_nonnegativity_violation_percent_mean")
        .reindex(index=MODEL_ORDER, columns=COMPONENTS)
        .to_numpy(float)
    )
    require(np.all(np.isfinite(matrix)), "Heatmap contains a non-finite value.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "nonnegativity",
        ["#F7F7F5", "#F4A261", "#E76F51", "#9B2226"],
    )
    color_max = 5.0 * np.ceil(float(matrix.max()) / 5.0)
    fig, axis = plt.subplots(figsize=(10.2, 5.3), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=color_max)
    axis.set_title(r"Component-level raw non-negativity violations at $N=10{,}000$")
    axis.set_xlabel("Effluent component")
    axis.set_xticks(range(len(COMPONENTS)), [COMPONENT_LABELS[name] for name in COMPONENTS], rotation=45, ha="right")
    axis.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[key] for key in MODEL_ORDER])
    colorbar = fig.colorbar(image, ax=axis, pad=0.025, shrink=0.90)
    colorbar.set_label("Five-fold mean raw violation (%)")

    fig.savefig(
        output_root / "figure_component_nonnegativity_violations_id.pdf",
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "Component non-negativity heatmap builder", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    print(f"Wrote heatmap assets to {output_root}")
    print(f"Maximum cell: {matrix.max():.2f}%")


if __name__ == "__main__":
    main()
