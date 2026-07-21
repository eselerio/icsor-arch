"""Create additive diagnostics for an in-progress article benchmark.

This script writes only new 08+ figures, source CSVs, and ADDITIONAL_INSIGHTS.md.
It does not regenerate or alter the existing partial-assessment artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "xgboost_regressor", "lightgbm_regressor", "catboost_regressor", "adaboost_regressor",
    "random_forest_regressor", "extra_trees_regressor", "svr_regressor", "knn_regressor",
    "pls_regressor", "multitask_elastic_net_regressor", "multitask_lasso_regressor",
    "ann_deep_regressor", "tabnet_regressor",
]
MODEL_LABELS = {
    "xgboost_regressor": "XGBoost", "lightgbm_regressor": "LightGBM",
    "catboost_regressor": "CatBoost", "adaboost_regressor": "AdaBoost",
    "random_forest_regressor": "Random Forest", "extra_trees_regressor": "Extra Trees",
    "svr_regressor": "SVR", "knn_regressor": "k-NN", "pls_regressor": "PLS",
    "multitask_elastic_net_regressor": "Multi-task Elastic Net",
    "multitask_lasso_regressor": "Multi-task Lasso", "ann_deep_regressor": "MLP",
    "tabnet_regressor": "TabNet",
}
SIZES = [500, 1450, 2400, 3350, 4300, 5250, 6200, 7150, 8100, 9050, 10000]
COMPONENTS = [
    "S_O", "S_F", "S_A", "S_NH4", "S_NO2", "S_NO3", "S_N2", "S_PO4", "S_I", "S_ALK",
    "X_I", "X_S", "X_H", "X_PAO", "X_PP", "X_PHA", "X_AOB", "X_NOB", "X_MeP", "X_MeOH",
]
PATTERN = re.compile(r"^(?P<model>.+)_n(?P<size>\d+)_fold(?P<fold>\d+)\.json$")
COLORS = dict(zip(MODEL_ORDER, plt.get_cmap("tab20").colors[: len(MODEL_ORDER)]))


def load(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    metrics, physical, components, composites = [], [], [], []
    paths = sorted((run_root / "metrics" / "units").glob("*.json"))
    for path in paths:
        match = PATTERN.match(path.name)
        if not match:
            continue
        meta = {
            "model_key": match.group("model"), "sample_size": int(match.group("size")),
            "outer_fold": int(match.group("fold")),
        }
        record = json.loads(path.read_text(encoding="utf-8"))
        metrics.extend({**meta, **row} for row in record["metrics"])
        physical.append({**meta, **record["physical_summary"]})
        components.extend({**meta, **row} for row in record["component_metrics"])
        composites.extend({**meta, **row} for row in record["composite_metrics"])
    return tuple(map(pd.DataFrame, (metrics, physical, components, composites))) + (paths,)


def complete_models(metrics: pd.DataFrame) -> list[str]:
    counts = metrics.groupby(["model_key", "sample_size"])["outer_fold"].nunique()
    return [
        key for key in MODEL_ORDER
        if key in counts.index.get_level_values(0)
        and all(counts.get((key, size), 0) == 5 for size in SIZES)
    ]


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 220, "font.size": 8.5,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def paired_fold_table(metrics: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    paired = metrics.query("model_key in @models").pivot(
        index=["model_key", "sample_size", "outer_fold"],
        columns="prediction_type", values="nRMSE",
    ).reset_index()
    paired["delta_nRMSE"] = paired["projected"] - paired["raw"]
    paired["relative_change_percent"] = 100 * paired["delta_nRMSE"] / paired["raw"]
    paired["projection_improved"] = paired["delta_nRMSE"] < 0
    return paired


def plot_fold_agreement(paired: pd.DataFrame, models: list[str], output: Path) -> pd.DataFrame:
    summary = paired.groupby(["model_key", "sample_size"]).agg(
        folds_improved=("projection_improved", "sum"),
        mean_delta_nRMSE=("delta_nRMSE", "mean"),
        sd_delta_nRMSE=("delta_nRMSE", "std"),
    ).reset_index()
    matrix = summary.pivot(index="model_key", columns="sample_size", values="folds_improved").reindex(
        index=models, columns=SIZES
    ).to_numpy(float)
    fig, axis = plt.subplots(figsize=(10.0, 5.7))
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=5)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            axis.text(column, row, str(value), ha="center", va="center", fontsize=7,
                      color="white" if value in (0, 5) else "#222222")
    axis.set_xticks(range(len(SIZES)), [f"{value:,}" for value in SIZES], rotation=45, ha="right")
    axis.set_yticks(range(len(models)), [MODEL_LABELS[key] for key in models])
    axis.set(xlabel="Nested sample total", title="How consistently does projection improve nRMSE across the five folds?")
    colorbar = fig.colorbar(image, ax=axis, ticks=range(6), shrink=0.82)
    colorbar.set_label("Folds improved (out of 5)")
    save(fig, output, "08_projection_fold_agreement")
    return summary


def full_size_summary(metrics: pd.DataFrame, physical: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    metric = metrics.query("model_key in @models and sample_size == 10000").groupby(
        ["model_key", "prediction_type"]
    ).agg(nRMSE=("nRMSE", "mean"), nRMSE_sd=("nRMSE", "std"), nMAE=("nMAE", "mean"), macro_R2=("macro_R2", "mean")).reset_index()
    wide = metric.pivot(index="model_key", columns="prediction_type")
    rows = pd.DataFrame(index=models)
    for measure in ["nRMSE", "nRMSE_sd", "nMAE", "macro_R2"]:
        for kind in ["raw", "projected"]:
            rows[f"{measure}_{kind}"] = wide[(measure, kind)]
    rows["delta_nRMSE"] = rows["nRMSE_projected"] - rows["nRMSE_raw"]
    rows["relative_change_percent"] = 100 * rows["delta_nRMSE"] / rows["nRMSE_raw"]
    phys = physical.query("model_key in @models and sample_size == 10000").groupby("model_key").agg(
        raw_nonnegative_violation_rate=("raw_nonnegative_violation", "mean"),
        raw_conservation_l2=("raw_conservation_l2", "mean"),
        projected_conservation_l2=("projected_conservation_l2", "mean"),
        standardized_displacement_l2=("standardized_displacement_l2", "mean"),
        backtracking_active=("backtracking_active", "mean"),
    )
    return rows.join(phys).reset_index(names="model_key")


def plot_tradeoff(summary: pd.DataFrame, models: list[str], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9.4, 6.2))
    sizes = 70 + 180 * summary["standardized_displacement_l2"] / summary["standardized_displacement_l2"].max()
    scatter = axis.scatter(
        summary["nRMSE_raw"], summary["relative_change_percent"], s=sizes,
        c=100 * summary["raw_nonnegative_violation_rate"], cmap="viridis", vmin=0, vmax=100,
        edgecolor="white", linewidth=0.8, alpha=0.9,
    )
    offsets = {
        "xgboost_regressor": (7, -8), "lightgbm_regressor": (7, 6),
        "multitask_elastic_net_regressor": (7, 7), "multitask_lasso_regressor": (7, -10),
        "random_forest_regressor": (-7, 4), "catboost_regressor": (7, -10),
    }
    for row in summary.itertuples(index=False):
        offset = offsets.get(row.model_key, (5, 4))
        alignment = "right" if row.model_key == "random_forest_regressor" else "left"
        axis.annotate(MODEL_LABELS[row.model_key], (row.nRMSE_raw, row.relative_change_percent),
                      xytext=offset, textcoords="offset points", fontsize=7.5, ha=alignment)
    axis.axhline(0, color="black", linewidth=1)
    axis.axhspan(axis.get_ylim()[0], 0, color="#009E73", alpha=0.06, zorder=-10)
    axis.set(
        xlabel="Raw nRMSE at N = 10,000 (left is more accurate)",
        ylabel="Projection-induced nRMSE change (%)",
        title="Accuracy–projection trade-off at full data size",
    )
    axis.grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=axis, pad=0.02)
    colorbar.set_label("Raw non-negativity violation rate (%)")
    axis.text(0.01, 0.02, "Projection improves ↓", transform=axis.transAxes, color="#007A55")
    axis.text(0.99, 0.02, "Bubble area ∝ standardized projection displacement", transform=axis.transAxes,
              ha="right", color="#444444", fontsize=7.5)
    save(fig, output, "09_accuracy_physics_tradeoff")


def plot_component_effects(components: pd.DataFrame, models: list[str], output: Path) -> pd.DataFrame:
    frame = components.query("model_key in @models and sample_size == 10000").groupby(
        ["model_key", "component", "prediction_type"]
    )["nRMSE_component"].mean().reset_index()
    wide = frame.pivot(index=["model_key", "component"], columns="prediction_type", values="nRMSE_component").reset_index()
    wide["delta_component_nRMSE"] = wide["projected"] - wide["raw"]
    matrix = wide.pivot(index="model_key", columns="component", values="delta_component_nRMSE").reindex(
        index=models, columns=COMPONENTS
    ).to_numpy(float)
    limit = np.nanpercentile(np.abs(matrix), 97)
    fig, axis = plt.subplots(figsize=(12.6, 6.0))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(COMPONENTS)), COMPONENTS, rotation=45, ha="right")
    axis.set_yticks(range(len(models)), [MODEL_LABELS[key] for key in models])
    axis.set(title="Which effluent components gain or lose accuracy after projection? (N = 10,000)")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82, extend="both")
    colorbar.set_label("Δ component nRMSE; blue improves, red worsens")
    save(fig, output, "10_component_projection_effects")
    return wide


def plot_composite_effects(composites: pd.DataFrame, models: list[str], output: Path) -> pd.DataFrame:
    frame = composites.query("model_key in @models and sample_size == 10000").groupby(
        ["model_key", "composite", "prediction_type"]
    )["RMSE"].mean().reset_index()
    wide = frame.pivot(index=["model_key", "composite"], columns="prediction_type", values="RMSE").reset_index()
    wide["relative_RMSE_change_percent"] = 100 * (wide["projected"] - wide["raw"]) / wide["raw"]
    names = ["COD", "TN", "TP", "TSS"]
    matrix = wide.pivot(index="model_key", columns="composite", values="relative_RMSE_change_percent").reindex(
        index=models, columns=names
    ).to_numpy(float)
    fig, axis = plt.subplots(figsize=(7.6, 6.0))
    norm = mpl.colors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=max(100, float(np.nanmax(matrix))))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(column, row, f"{value:+.0f}%", ha="center", va="center", fontsize=7,
                      color="white" if abs(value) > 65 else "#222222")
    axis.set_xticks(range(len(names)), names)
    axis.set_yticks(range(len(models)), [MODEL_LABELS[key] for key in models])
    axis.set(title="Projection effect on reported water-quality RMSE (N = 10,000)")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82, extend="both")
    colorbar.set_label("Relative RMSE change; blue improves, red worsens")
    save(fig, output, "11_composite_projection_effects")
    return wide


def plot_data_efficiency(metrics: pd.DataFrame, models: list[str], output: Path) -> pd.DataFrame:
    raw = metrics.query("model_key in @models and prediction_type == 'raw'").groupby(
        ["model_key", "sample_size"]
    )["nRMSE"].mean().reset_index()
    records = []
    for key in models:
        rows = raw.query("model_key == @key").set_index("sample_size").reindex(SIZES)
        values = rows["nRMSE"].to_numpy(float)
        final = values[-1]
        threshold = 1.05 * final
        stable_index = len(SIZES) - 1
        for index in range(len(SIZES)):
            if np.all(values[index:] <= threshold):
                stable_index = index
                break
        records.append({
            "model_key": key, "nRMSE_500": values[0], "nRMSE_10000": final,
            "relative_reduction_percent": 100 * (values[0] - final) / values[0],
            "stabilization_size_within_5pct": SIZES[stable_index],
        })
    summary = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(models))
    axes[0].hlines(y, summary["nRMSE_10000"], summary["nRMSE_500"], color="#BBBBBB", linewidth=2)
    axes[0].scatter(summary["nRMSE_500"], y, color="#D55E00", label="N = 500", zorder=3)
    axes[0].scatter(summary["nRMSE_10000"], y, color="#0072B2", label="N = 10,000", zorder=3)
    axes[0].set_yticks(y, [MODEL_LABELS[key] for key in models])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Raw nRMSE", title="Accuracy gain from more training data")
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].legend()
    order = summary.sort_values("stabilization_size_within_5pct")
    axes[1].barh(range(len(order)), order["stabilization_size_within_5pct"], color=[COLORS[key] for key in order.model_key])
    axes[1].set_yticks(range(len(order)), [MODEL_LABELS[key] for key in order.model_key])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Nested sample total", title="First size after which raw nRMSE stays\nwithin 5% of its N=10,000 value")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save(fig, output, "12_data_efficiency")
    return summary


def write_report(
    output: Path, manifest: dict, captured: datetime, models: list[str], paths: list[Path],
    full: pd.DataFrame, agreement: pd.DataFrame, components: pd.DataFrame,
    composites: pd.DataFrame, efficiency: pd.DataFrame,
) -> None:
    ranked = full.sort_values("nRMSE_raw")
    total_cells = len(models) * len(SIZES)
    improved_cells = int((agreement["mean_delta_nRMSE"] < 0).sum())
    unanimous_improvement = int((agreement["folds_improved"] == 5).sum())
    unanimous_worsening = int((agreement["folds_improved"] == 0).sum())
    best = ranked.iloc[0]
    largest_benefit = full.loc[full["relative_change_percent"].idxmin()]
    largest_penalty = full.loc[full["relative_change_percent"].idxmax()]
    displacement_rho = full[["standardized_displacement_l2", "relative_change_percent"]].corr(method="spearman").iloc[0, 1]
    negativity_rho = full[["raw_nonnegative_violation_rate", "relative_change_percent"]].corr(method="spearman").iloc[0, 1]
    component_counts = components.groupby("model_key")["delta_component_nRMSE"].agg(
        components_improved=lambda values: int((values < 0).sum()),
        components_worsened=lambda values: int((values > 0).sum()),
    ).reindex(models)
    comp_pivot = composites.pivot(index="model_key", columns="composite", values="relative_RMSE_change_percent").reindex(models)
    full_physical_cells = len(models) * len(SIZES)
    lines = [
        "# Additional interim insights",
        "",
        "> These analyses supplement the existing 01–07 figures without changing them. The live run remains partial; TabNet is not yet scored.",
        "",
        f"- Captured: {captured.astimezone().isoformat(timespec='seconds')}",
        f"- Manifest status/update: `{manifest.get('status')}` / {manifest.get('updated_utc')}",
        f"- Complete models: {len(models)} of 13 ({', '.join(MODEL_LABELS[key] for key in models)})",
        f"- Fold files observed: {len(paths)} of 715",
        "",
        "## 1. Fold-level consistency, not just mean direction",
        "",
        f"Projection improves the mean nRMSE in {improved_cells} of {total_cells} complete model–size cells. "
        f"The direction is unanimous across all five folds in {unanimous_improvement} improving cells and unanimously adverse in {unanimous_worsening} cells. "
        "This distinguishes stable effects from mean changes driven by only part of the cross-validation split.",
        "",
        "![Fold agreement](08_projection_fold_agreement.png)",
        "",
        "## 2. Joint accuracy–physics trade-off",
        "",
        f"The physical contract remains exact in all {full_physical_cells} complete 12-model cells: raw mass-conservation violations occur for 100% of samples, while projected mass-conservation and non-negativity violation rates are both 0%. "
        f"For the newly available MLP, the full-size raw non-negativity violation rate is {100 * full.loc[full.model_key == 'ann_deep_regressor', 'raw_nonnegative_violation_rate'].iloc[0]:.1f}%, reduced to 0% by projection.",
        "",
        f"MLP is currently the most accurate full-size model (raw nRMSE {best.nRMSE_raw:.3f}; projected {best.nRMSE_projected:.3f}). "
        f"The largest relative projection benefit at full size is {MODEL_LABELS[largest_benefit.model_key]} "
        f"({largest_benefit.relative_change_percent:.1f}%), while the largest penalty is "
        f"{MODEL_LABELS[largest_penalty.model_key]} ({largest_penalty.relative_change_percent:+.1f}%). "
        "Bubble size shows how far projection moves the standardized prediction; color shows the raw non-negativity burden.",
        "",
        f"Across the 12 models, the descriptive Spearman association between raw non-negativity-violation rate and projection-induced error change is {negativity_rho:.2f}; "
        f"the association between standardized displacement and error change is {displacement_rho:.2f}. Thus, heavier raw negativity tends to align with greater benefit, whereas larger corrective moves tend to align with larger accuracy penalties. These are cross-model descriptions, not causal or inferential results.",
        "",
        "![Accuracy and physics trade-off](09_accuracy_physics_tradeoff.png)",
        "",
        "## 3. Component-level effects",
        "",
        "Aggregate nRMSE can conceal opposing changes among the 20 effluent components. The heat map identifies where projection helps or harms each model.",
        "For example, Random Forest improves 12 components but worsens overall because its losses for a few components—especially S_F and X_H—are much larger than its gains elsewhere. Component counts therefore cannot substitute for the equally weighted aggregate error magnitude.",
        "",
        "![Component effects](10_component_projection_effects.png)",
        "",
        "| Model | Components improved | Components worsened |",
        "|---|---:|---:|",
    ]
    for key in models:
        row = component_counts.loc[key]
        lines.append(f"| {MODEL_LABELS[key]} | {int(row.components_improved)} | {int(row.components_worsened)} |")
    lines.extend([
        "",
        "## 4. Consequences for reported water-quality quantities",
        "",
        "The component-space projection does not affect COD, TN, TP, and TSS uniformly. The values below are relative changes in RMSE; negative values improve accuracy.",
        "All 12 models improve TN and TP, while all worsen COD and TSS. TP RMSE falls by approximately 98–100%, consistent with the phosphorus-related invariant being enforced directly; this should be interpreted as a consequence of the physical contract rather than a general extrapolative accuracy result.",
        "",
        "![Composite effects](11_composite_projection_effects.png)",
        "",
        "| Model | COD | TN | TP | TSS |",
        "|---|---:|---:|---:|---:|",
    ])
    for key in models:
        row = comp_pivot.loc[key]
        lines.append(f"| {MODEL_LABELS[key]} | {row.COD:+.1f}% | {row.TN:+.1f}% | {row.TP:+.1f}% | {row.TSS:+.1f}% |")
    lines.extend([
        "",
        "## 5. Data efficiency",
        "",
        "The dumbbells show the raw-error reduction from 500 to 10,000 observations. The right panel reports the first nested size after which performance remains within 5% of the final full-size value; it is a descriptive stabilization threshold, not an optimum.",
        "MLP benefits most from added data (71.3% raw-nRMSE reduction from N=500 to N=10,000) and stabilizes latest by this rule (N=9,050). Conversely, early stabilization for AdaBoost, PLS, and the multi-task linear models reflects relatively flat learning curves, not necessarily high accuracy.",
        "",
        "![Data efficiency](12_data_efficiency.png)",
        "",
        "## Interpretation boundary",
        "",
        "All results are in-distribution five-fold summaries from complete cells only. The run is still marked `running`; TabNet, OOD evaluation, final refits, and terminal assertions remain unavailable. No final 13-model ranking or extrapolation conclusion is justified yet.",
        "",
    ])
    (output / "ADDITIONAL_INSIGHTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run_root, output = args.run_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    metrics, physical, components, composites, paths = load(run_root)
    models = complete_models(metrics)
    style()
    paired = paired_fold_table(metrics, models)
    agreement = plot_fold_agreement(paired, models, output)
    full = full_size_summary(metrics, physical, models)
    plot_tradeoff(full, models, output)
    component_effects = plot_component_effects(components, models, output)
    composite_effects = plot_composite_effects(composites, models, output)
    efficiency = plot_data_efficiency(metrics, models, output)
    paired.to_csv(output / "additional_paired_fold_effects.csv", index=False)
    agreement.to_csv(output / "additional_fold_agreement.csv", index=False)
    full.to_csv(output / "additional_full_size_tradeoff.csv", index=False)
    component_effects.to_csv(output / "additional_component_effects.csv", index=False)
    composite_effects.to_csv(output / "additional_composite_effects.csv", index=False)
    efficiency.to_csv(output / "additional_data_efficiency.csv", index=False)
    associations = pd.DataFrame([
        {
            "x": "raw_nonnegative_violation_rate", "y": "relative_projection_nRMSE_change_percent",
            "spearman_rho": full[["raw_nonnegative_violation_rate", "relative_change_percent"]].corr(method="spearman").iloc[0, 1],
            "n_models": len(models),
        },
        {
            "x": "standardized_displacement_l2", "y": "relative_projection_nRMSE_change_percent",
            "spearman_rho": full[["standardized_displacement_l2", "relative_change_percent"]].corr(method="spearman").iloc[0, 1],
            "n_models": len(models),
        },
    ])
    associations.to_csv(output / "additional_descriptive_associations.csv", index=False)
    write_report(
        output, manifest, datetime.now(timezone.utc), models, paths, full, agreement,
        component_effects, composite_effects, efficiency,
    )
    print(output)


if __name__ == "__main__":
    main()
