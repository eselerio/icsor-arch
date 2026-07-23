from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "article_final_v2"
OUTPUT = ROOT / "article" / "wip"

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
LATEX_MODEL_LABELS = {**MODEL_LABELS, "knn_regressor": "$k$-NN"}
MODEL_COLORS = {
    "xgboost_regressor": "#577590",
    "lightgbm_regressor": "#264653",
    "catboost_regressor": "#E76F51",
    "adaboost_regressor": "#F4A261",
    "random_forest_regressor": "#8D99AE",
    "extra_trees_regressor": "#6B705C",
    "svr_regressor": "#BC4749",
    "knn_regressor": "#E9C46A",
    "pls_regressor": "#5C6770",
    "multitask_elastic_net_regressor": "#2A9D8F",
    "multitask_lasso_regressor": "#4D908E",
    "ann_deep_regressor": "#6D597A",
    "tabnet_regressor": "#9C6644",
}
MODEL_MARKERS = {
    key: marker
    for key, marker in zip(
        MODEL_ORDER,
        ["o", "s", "^", "v", "D", "d", "P", ">", "<", "h", "+", "p", "*"],
        strict=True,
    )
}
SAMPLE_SIZES = [500, 1450, 2400, 3350, 4300, 5250, 6200, 7150, 8100, 9050, 10000]
COMPONENT_ORDER = [
    "S_O", "S_F", "S_A", "S_NH4", "S_NO2", "S_NO3", "S_N2", "S_PO4", "S_I", "S_ALK",
    "X_I", "X_S", "X_H", "X_PAO", "X_PP", "X_PHA", "X_AOB", "X_NOB", "X_MeP", "X_MeOH",
]
COMPONENT_LABELS = {
    component: "$" + component.replace("_", "_{", 1) + "}$" if "_" in component else component
    for component in COMPONENT_ORDER
}
COMPOSITE_ORDER = ["COD", "TN", "TP", "TSS"]
UNIT_LABELS = {
    "g O2/m^3": "g O$_2$ m$^{-3}$",
    "g COD/m^3": "g COD m$^{-3}$",
    "g N/m^3": "g N m$^{-3}$",
    "g P/m^3": "g P m$^{-3}$",
    "g TSS/m^3": "g TSS m$^{-3}$",
    "mol/m^3": "mol m$^{-3}$",
}


def require_completed_run() -> None:
    completed_path = RUN_ROOT / "COMPLETED.json"
    manifest_path = RUN_ROOT / "manifest.json"
    if not completed_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"The completed run is missing from {RUN_ROOT}.")
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if completed.get("run_id") != "article_final_v2" or completed.get("profile") != "full":
        raise RuntimeError("The completion marker is not the full article_final_v2 run.")
    if not completed.get("article_eligible") or manifest.get("status") != "complete":
        raise RuntimeError("The run is not complete and article-eligible.")


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT / f"{stem}.pdf",
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "Manuscript revision asset builder", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def mean_sd(values: Iterable[float], digits: int = 3) -> str:
    series = pd.Series(list(values), dtype=float)
    return f"${series.mean():.{digits}f}\\pm{series.std(ddof=1):.{digits}f}$"


def signed_mean_sd(values: Iterable[float], digits: int = 3) -> str:
    series = pd.Series(list(values), dtype=float)
    return f"${series.mean():+.{digits}f}\\pm{series.std(ddof=1):.{digits}f}$"


def longtable(
    path: Path,
    caption: str,
    label: str,
    column_spec: str,
    headings: list[str],
    rows: list[list[str]],
) -> None:
    del caption, label, column_spec
    columns = [f"col{chr(ord('A') + index)}" for index in range(len(headings))]
    csv_rows = [
        [
            re.sub(r"(?<!\\),", r"\\csvcomma{}", str(value).replace('"', r"\textquotedbl{}"))
            for value in row
        ]
        for row in rows
    ]
    pd.DataFrame(csv_rows, columns=columns).to_csv(path, index=False)


def plot_model_curve(
    axis: mpl.axes.Axes,
    rows: pd.DataFrame,
    mean_column: str,
    sd_column: str,
) -> None:
    for model_key in MODEL_ORDER:
        subset = rows.query("model_key == @model_key").set_index("sample_size").reindex(SAMPLE_SIZES)
        mean = subset[mean_column].to_numpy(float)
        sd = subset[sd_column].fillna(0.0).to_numpy(float)
        axis.plot(
            SAMPLE_SIZES,
            mean,
            color=MODEL_COLORS[model_key],
            marker=MODEL_MARKERS[model_key],
            linewidth=1.25,
            markersize=4,
            label=MODEL_LABELS[model_key],
        )
        lower = np.maximum(mean - sd, np.finfo(float).tiny) if axis.get_yscale() == "log" else mean - sd
        axis.fill_between(
            SAMPLE_SIZES,
            lower,
            mean + sd,
            color=MODEL_COLORS[model_key],
            alpha=0.07,
            linewidth=0,
        )


def plot_learning_curves(id_summary: pd.DataFrame) -> None:
    raw = id_summary.query("prediction_type == 'raw'").copy()
    figure, axis = plt.subplots(figsize=(9.3, 5.7))
    plot_model_curve(axis, raw, "nRMSE_mean", "nRMSE_std")
    axis.set(
        xlabel="Nested sample total",
        ylabel="nRMSE",
        title="In-distribution raw-prediction learning curves",
    )
    axis.set_xticks(SAMPLE_SIZES, [f"{value:,}" for value in SAMPLE_SIZES], rotation=45, ha="right")
    axis.grid(axis="y", linestyle=":", alpha=0.3)
    axis.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(figure, "figure_learning_curves")
    raw.to_csv(OUTPUT / "source_learning_curves.csv", index=False)


def plot_projection_change(id_summary: pd.DataFrame) -> None:
    pivot = id_summary.pivot(
        index=["model", "model_key", "sample_size"],
        columns="prediction_type",
        values="nRMSE_mean",
    ).reset_index()
    pivot["delta_nRMSE"] = pivot["projected"] - pivot["raw"]
    matrix = np.vstack(
        [
            pivot.query("model_key == @model_key").set_index("sample_size").reindex(SAMPLE_SIZES)["delta_nRMSE"].to_numpy(float)
            for model_key in MODEL_ORDER
        ]
    )
    limit = max(float(np.nanmax(np.abs(matrix))), np.finfo(float).eps)
    figure, axis = plt.subplots(figsize=(10.2, 5.3), constrained_layout=True)
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        norm=mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_title("In-distribution effect of projection across sample size")
    axis.set_xlabel("Nested sample total")
    axis.set_xticks(range(len(SAMPLE_SIZES)), [f"{value:,}" for value in SAMPLE_SIZES], rotation=45, ha="right")
    axis.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[key] for key in MODEL_ORDER])
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, shrink=0.88)
    colorbar.set_label(r"$\Delta$nRMSE (projected $-$ raw; linear scale)")
    save_figure(figure, "figure_projection_change")
    pivot.to_csv(OUTPUT / "source_projection_change.csv", index=False)


def plot_inference_latency(timing: pd.DataFrame) -> None:
    rows = timing.query("sample_size == 10000").set_index("model_key").reindex(MODEL_ORDER)
    positions = np.arange(len(MODEL_ORDER))
    figure, axis = plt.subplots(figsize=(9.3, 5.7))
    axis.bar(
        positions,
        rows["raw_latency_ms_per_sample_mean"].to_numpy(float),
        yerr=rows["raw_latency_ms_per_sample_std"].to_numpy(float),
        color=[MODEL_COLORS[key] for key in MODEL_ORDER],
        edgecolor="white",
        linewidth=0.6,
        capsize=2.5,
        error_kw={"ecolor": "#5C6770", "elinewidth": 0.9},
    )
    axis.set(
        ylabel="Raw inference latency (ms per sample)",
        title=r"Complete 20-component inference at $N=10{,}000$",
    )
    axis.set_xticks(positions, [MODEL_LABELS[key] for key in MODEL_ORDER], rotation=45, ha="right")
    axis.grid(axis="y", linestyle=":", alpha=0.3)
    figure.tight_layout()
    save_figure(figure, "figure_inference_latency")
    rows.reset_index().to_csv(OUTPUT / "source_inference_latency.csv", index=False)


def plot_evaluation_workflow() -> None:
    figure, axis = plt.subplots(figsize=(11.0, 4.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def box(x: float, y: float, width: float, height: float, label: str, projection: bool = False) -> dict[str, tuple[float, float]]:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#F8E8B6" if projection else "#F2F8F7",
            edgecolor="#E76F51" if projection else "#264653",
            linewidth=1.4,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9.5)
        return {
            "left": (x, y + height / 2),
            "right": (x + width, y + height / 2),
            "top": (x + width / 2, y + height),
            "bottom": (x + width / 2, y),
        }

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.5,
                color="#2A9D8F",
                shrinkA=2,
                shrinkB=2,
            )
        )

    data = box(0.02, 0.62, 0.20, 0.22, "Mechanistic states\n22 inputs / 20 targets")
    model = box(0.27, 0.62, 0.20, 0.22, "Fold-local preprocessing\n13 statistical surrogates")
    raw = box(0.52, 0.62, 0.18, 0.22, "Raw physical\n20-component prediction")
    projected = box(0.76, 0.62, 0.21, 0.22, "Kircher--Votsmeier\nprojection", projection=True)
    extrapolation = box(0.02, 0.14, 0.25, 0.22, "13 full-data refits\n" + r"$\rightarrow$ untouched extrapolation set")
    score = box(0.48, 0.10, 0.40, 0.30, "Paired accuracy, mass conservation,\nnon-negativity, COD/TN/TP/TSS,\ndisplacement, and latency")

    arrow(data["right"], model["left"])
    arrow(model["right"], raw["left"])
    arrow(raw["right"], projected["left"])
    arrow(raw["bottom"], score["top"])
    arrow(projected["bottom"], score["right"])
    arrow(data["bottom"], extrapolation["top"])
    arrow(extrapolation["right"], score["left"])
    figure.tight_layout()
    save_figure(figure, "figure_evaluation_workflow")


def plot_violation_rates(physical: pd.DataFrame) -> None:
    raw = physical.query("prediction_type == 'raw'").copy()
    figure, axes = plt.subplots(2, 1, figsize=(9.3, 7.4), sharex=True, gridspec_kw={"height_ratios": [0.72, 1.55]})

    axes[0].plot(SAMPLE_SIZES, np.full(len(SAMPLE_SIZES), 100.0), color="#BC4749", linewidth=2.1, label="Raw (all 13 surrogates)")
    axes[0].plot(SAMPLE_SIZES, np.zeros(len(SAMPLE_SIZES)), color="#2A9D8F", linewidth=2.1, linestyle="--", label="Projected (all 13 surrogates)")
    axes[0].set(ylabel="Rows violating mass conservation (%)", title="(a) Mass-conservation violations")
    axes[0].set_ylim(-4, 104)
    axes[0].legend(frameon=False, ncol=2, loc="center")
    axes[0].grid(axis="y", linestyle=":", alpha=0.3)

    for model_key in MODEL_ORDER:
        rows = raw.query("model_key == @model_key").set_index("sample_size").reindex(SAMPLE_SIZES)
        axes[1].plot(
            SAMPLE_SIZES,
            100.0 * rows["nonnegative_violation_rate_mean"].to_numpy(float),
            color=MODEL_COLORS[model_key],
            marker=MODEL_MARKERS[model_key],
            linewidth=1.25,
            markersize=4,
            label=MODEL_LABELS[model_key],
        )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.1, label="Projected (all surrogates)")
    axes[1].set(
        xlabel="Nested sample total",
        ylabel="Rows violating non-negativity (%)",
        title="(b) Non-negativity violations",
    )
    axes[1].set_ylim(-4, 104)
    axes[1].set_xticks(SAMPLE_SIZES, [f"{value:,}" for value in SAMPLE_SIZES], rotation=45, ha="right")
    axes[1].grid(axis="y", linestyle=":", alpha=0.3)
    axes[1].legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    figure.suptitle("In-distribution mass-conservation and non-negativity violations across sample size", y=0.995)
    figure.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_figure(figure, "figure_violation_rates_by_sample_size")

    raw[[
        "model", "model_key", "sample_size", "mass_violation_rate_mean",
        "nonnegative_violation_rate_mean", "negative_standardized_l1_mean_mean",
    ]].to_csv(OUTPUT / "source_violation_rates_by_sample_size.csv", index=False)


def component_delta_frame(component_metrics: pd.DataFrame, query: str) -> pd.DataFrame:
    frame = component_metrics.query(query).groupby(
        [column for column in ("regime", "model_key", "component", "prediction_type") if column in component_metrics],
        sort=False,
    )["nRMSE_component"].mean().reset_index()
    index = [column for column in ("regime", "model_key", "component") if column in frame]
    wide = frame.pivot(index=index, columns="prediction_type", values="nRMSE_component").reset_index()
    wide["delta_component_nRMSE"] = wide["projected"] - wide["raw"]
    return wide


def heatmap_norm(values: np.ndarray) -> mpl.colors.SymLogNorm:
    limit = max(float(np.nanmax(np.abs(values))), 1e-6)
    return mpl.colors.SymLogNorm(linthresh=0.01, linscale=0.8, vmin=-limit, vmax=limit, base=10)


def plot_id_component_effects(component_metrics: pd.DataFrame) -> pd.DataFrame:
    delta = component_delta_frame(component_metrics, "sample_size == 10000")
    matrix = delta.pivot(index="model_key", columns="component", values="delta_component_nRMSE").reindex(
        index=MODEL_ORDER, columns=COMPONENT_ORDER
    ).to_numpy(float)
    figure, axis = plt.subplots(figsize=(10.2, 5.3), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=heatmap_norm(matrix))
    axis.set_xticks(range(len(COMPONENT_ORDER)), [COMPONENT_LABELS[name] for name in COMPONENT_ORDER], rotation=45, ha="right")
    axis.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[name] for name in MODEL_ORDER])
    axis.set(title="Component-level effect of projection at N = 10,000", xlabel="Effluent component")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.84, extend="both")
    colorbar.set_label("Projected minus raw component nRMSE (symmetric-log color scale)")
    save_figure(figure, "figure_component_projection_effects_id")
    delta.to_csv(OUTPUT / "source_component_projection_effects_id.csv", index=False)
    return delta


def plot_training_time(timing: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9.3, 5.7))
    axis.set_yscale("log")
    plot_model_curve(axis, timing, "setup_seconds_mean", "setup_seconds_std")
    axis.set(xlabel="Nested sample total", ylabel="Preprocessing and fitting time (s, logarithmic scale)", title="Surrogate training time across sample size")
    axis.set_xticks(SAMPLE_SIZES, [f"{value:,}" for value in SAMPLE_SIZES], rotation=45, ha="right")
    axis.grid(axis="y", linestyle=":", alpha=0.3, which="both")
    axis.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(figure, "figure_training_time_scaling")
    timing.to_csv(OUTPUT / "source_training_time_scaling.csv", index=False)


def plot_ood_component_effects(ood_components: pd.DataFrame) -> pd.DataFrame:
    delta = component_delta_frame(ood_components, "regime in ['mild', 'severe']")
    matrices = {
        regime: delta.query("regime == @regime").pivot(index="model_key", columns="component", values="delta_component_nRMSE").reindex(
            index=MODEL_ORDER, columns=COMPONENT_ORDER
        ).to_numpy(float)
        for regime in ("mild", "severe")
    }
    combined = np.concatenate([matrices["mild"].ravel(), matrices["severe"].ravel()])
    norm = heatmap_norm(combined)
    figure, axes = plt.subplots(2, 1, figsize=(10.2, 8.3), sharex=True)
    images = []
    for axis, regime, panel in zip(axes, ("mild", "severe"), ("a", "b"), strict=True):
        images.append(axis.imshow(matrices[regime], aspect="auto", cmap="RdBu_r", norm=norm))
        axis.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[name] for name in MODEL_ORDER])
        axis.set_title(f"({panel}) {regime.capitalize()} extrapolation")
    axes[-1].set_xticks(range(len(COMPONENT_ORDER)), [COMPONENT_LABELS[name] for name in COMPONENT_ORDER], rotation=45, ha="right")
    axes[-1].set_xlabel("Effluent component")
    colorbar_axis = figure.add_axes([0.905, 0.17, 0.018, 0.66])
    colorbar = figure.colorbar(images[0], cax=colorbar_axis, extend="both")
    colorbar.set_label("Projected minus raw component nRMSE (symmetric-log color scale)")
    figure.suptitle("Component-level effect of projection beyond the training domain", y=0.995)
    figure.subplots_adjust(left=0.16, right=0.87, top=0.94, bottom=0.10, hspace=0.16)
    save_figure(figure, "figure_ood_component_projection_effects")
    delta.to_csv(OUTPUT / "source_ood_component_projection_effects.csv", index=False)
    return delta


def paired_id_rows(frame: pd.DataFrame, item_column: str, metric: str, delta_metric: str) -> pd.DataFrame:
    keys = ["model_key", item_column, "outer_fold"]
    selected = frame.query("sample_size == 10000") if "sample_size" in frame else frame
    wide = selected.pivot(index=keys, columns="prediction_type", values=[metric, "R2"]).reset_index()
    wide.columns = [column if isinstance(column, str) else "_".join(part for part in column if part) for column in wide.columns]
    wide[delta_metric] = wide[f"{metric}_projected"] - wide[f"{metric}_raw"]
    return wide


def write_accuracy_tables(
    id_components: pd.DataFrame,
    id_composites: pd.DataFrame,
    ood_components: pd.DataFrame,
    ood_composites: pd.DataFrame,
) -> None:
    id_component_wide = paired_id_rows(id_components, "component", "nRMSE_component", "delta")
    component_units = id_components.drop_duplicates("component").set_index("component")["unit"].to_dict()
    rows: list[list[str]] = []
    for component in COMPONENT_ORDER:
        for model_key in MODEL_ORDER:
            group = id_component_wide.query("component == @component and model_key == @model_key")
            rows.append([
                COMPONENT_LABELS[component],
                UNIT_LABELS[component_units[component]],
                LATEX_MODEL_LABELS[model_key],
                mean_sd(group["nRMSE_component_raw"]),
                mean_sd(group["nRMSE_component_projected"]),
                signed_mean_sd(group["delta"]),
                mean_sd(group["R2_raw"]),
                mean_sd(group["R2_projected"]),
            ])
    longtable(
        OUTPUT / "table_s_id_component_accuracy.csv",
        "Full-size in-distribution accuracy by effluent component and surrogate. Values are five-fold mean $\\pm$ sample SD; $\\Delta$ is projected minus raw on paired folds.",
        "tab:s_id_component_accuracy",
        "llp{3.1cm}rrrrr",
        ["Component", "Unit", "Surrogate", "Raw nRMSE$_j$", "Projected nRMSE$_j$", "$\\Delta$nRMSE$_j$", "Raw $R^2_j$", "Projected $R^2_j$"],
        rows,
    )

    id_composite_wide = paired_id_rows(id_composites, "composite", "nRMSE_composite", "delta")
    rows = []
    for composite in COMPOSITE_ORDER:
        for model_key in MODEL_ORDER:
            group = id_composite_wide.query("composite == @composite and model_key == @model_key")
            rows.append([
                composite,
                LATEX_MODEL_LABELS[model_key],
                mean_sd(group["nRMSE_composite_raw"]),
                mean_sd(group["nRMSE_composite_projected"]),
                signed_mean_sd(group["delta"]),
                mean_sd(group["R2_raw"]),
                mean_sd(group["R2_projected"]),
            ])
    longtable(
        OUTPUT / "table_s_id_composite_accuracy.csv",
        "Full-size in-distribution accuracy by derived water-quality composite and surrogate. Composite nRMSE is physical-unit RMSE divided by the population SD of that composite in the applicable outer-training partition. Values are five-fold mean $\\pm$ sample SD and $\\Delta$ is projected minus raw on paired folds.",
        "tab:s_id_composite_accuracy",
        "lp{3.1cm}rrrrr",
        ["Composite", "Surrogate", "Raw nRMSE$_q$", "Projected nRMSE$_q$", "$\\Delta$nRMSE$_q$", "Raw $R^2$", "Projected $R^2$"],
        rows,
    )

    for regime in ("mild", "severe"):
        current = ood_components.query("regime == @regime")
        wide = current.pivot(index=["model_key", "component"], columns="prediction_type", values=["nRMSE_component", "R2"]).reset_index()
        wide.columns = [column if isinstance(column, str) else "_".join(part for part in column if part) for column in wide.columns]
        wide["delta"] = wide["nRMSE_component_projected"] - wide["nRMSE_component_raw"]
        rows = []
        for component in COMPONENT_ORDER:
            for model_key in MODEL_ORDER:
                row = wide.query("component == @component and model_key == @model_key").iloc[0]
                rows.append([
                    COMPONENT_LABELS[component], UNIT_LABELS[component_units[component]], LATEX_MODEL_LABELS[model_key],
                    f"${row.nRMSE_component_raw:.3f}$", f"${row.nRMSE_component_projected:.3f}$", f"${row.delta:+.3f}$",
                    f"${row.R2_raw:.3f}$", f"${row.R2_projected:.3f}$",
                ])
        longtable(
            OUTPUT / f"table_s_ood_{regime}_component_accuracy.csv",
            f"{regime.capitalize()} out-of-distribution accuracy by effluent component and surrogate. Each value summarizes the same 600 mechanistic cases; $\\Delta$ is projected minus raw.",
            f"tab:s_ood_{regime}_component_accuracy",
            "llp{3.1cm}rrrrr",
            ["Component", "Unit", "Surrogate", "Raw nRMSE$_j$", "Projected nRMSE$_j$", "$\\Delta$nRMSE$_j$", "Raw $R^2_j$", "Projected $R^2_j$"],
            rows,
        )

        current = ood_composites.query("regime == @regime")
        wide = current.pivot(index=["model_key", "composite"], columns="prediction_type", values=["nRMSE_composite", "R2"]).reset_index()
        wide.columns = [column if isinstance(column, str) else "_".join(part for part in column if part) for column in wide.columns]
        wide["delta"] = wide["nRMSE_composite_projected"] - wide["nRMSE_composite_raw"]
        rows = []
        for composite in COMPOSITE_ORDER:
            for model_key in MODEL_ORDER:
                row = wide.query("composite == @composite and model_key == @model_key").iloc[0]
                rows.append([
                    composite, LATEX_MODEL_LABELS[model_key],
                    f"${row.nRMSE_composite_raw:.3f}$", f"${row.nRMSE_composite_projected:.3f}$", f"${row.delta:+.3f}$",
                    f"${row.R2_raw:.3f}$", f"${row.R2_projected:.3f}$",
                ])
        longtable(
            OUTPUT / f"table_s_ood_{regime}_composite_accuracy.csv",
            f"{regime.capitalize()} out-of-distribution accuracy by derived water-quality composite and surrogate. Composite nRMSE is physical-unit RMSE divided by the population SD of that composite in all 10,000 in-distribution targets. Each value summarizes the same 600 mechanistic cases; $\\Delta$ is projected minus raw.",
            f"tab:s_ood_{regime}_composite_accuracy",
            "lp{3.1cm}rrrrr",
            ["Composite", "Surrogate", "Raw nRMSE$_q$", "Projected nRMSE$_q$", "$\\Delta$nRMSE$_q$", "Raw $R^2$", "Projected $R^2$"],
            rows,
        )

    id_component_wide.to_csv(OUTPUT / "source_table_s_id_component_accuracy.csv", index=False)
    id_composite_wide.to_csv(OUTPUT / "source_table_s_id_composite_accuracy.csv", index=False)


def write_backtracking_table(physical_fold: pd.DataFrame) -> None:
    raw = physical_fold.query("prediction_type == 'raw'").copy()
    raw["backtracking_count"] = raw["backtracking_rate"] * raw["sample_size"] / 5.0
    totals = raw.groupby("model_key", sort=False)["backtracking_count"].sum()
    full = raw.query("sample_size == 10000").groupby("model_key", sort=False).agg(
        full_rate=("backtracking_rate", "mean"),
        full_displacement=("standardized_displacement_l2_mean", "mean"),
    )
    rows = []
    for model_key in MODEL_ORDER:
        count = int(round(totals.loc[model_key]))
        rows.append([
            LATEX_MODEL_LABELS[model_key],
            f"{count:,}",
            f"${100.0 * count / sum(SAMPLE_SIZES):.3f}$",
            f"${100.0 * full.loc[model_key, 'full_rate']:.3f}$",
            f"${full.loc[model_key, 'full_displacement']:.3f}$",
        ])
    longtable(
        OUTPUT / "table_s_backtracking.csv",
        "Positivity-backtracking activation by surrogate. The all-size denominator is 57,750 projections per surrogate; the final two columns describe $N=10{,}000$.",
        "tab:s_backtracking",
        "lrrrr",
        ["Surrogate", "All-size count", "All-size rate (\\%)", "Full-size rate (\\%)", "Full-size mean $d_c^{\\mathrm{std}}$"],
        rows,
    )


def write_timing_table(timing: pd.DataFrame) -> None:
    rows = []
    for model_key in MODEL_ORDER:
        at_500 = timing.query("model_key == @model_key and sample_size == 500").iloc[0]
        at_full = timing.query("model_key == @model_key and sample_size == 10000").iloc[0]
        ratio = at_full.setup_seconds_mean / at_500.setup_seconds_mean
        rows.append([
            LATEX_MODEL_LABELS[model_key],
            f"${at_500.setup_seconds_mean:.3f}\\pm{at_500.setup_seconds_std:.3f}$",
            f"${at_full.setup_seconds_mean:.3f}\\pm{at_full.setup_seconds_std:.3f}$",
            f"${ratio:.1f}$",
            f"${at_full.raw_latency_ms_per_sample_mean:.3f}\\pm{at_full.raw_latency_ms_per_sample_std:.3f}$",
            f"${at_full.projection_latency_ms_per_sample_mean:.3f}\\pm{at_full.projection_latency_ms_per_sample_std:.3f}$",
        ])
    longtable(
        OUTPUT / "table_s_timing.csv",
        "Computational timing summary. Fit time includes fold-local preprocessing and surrogate fitting but excludes hyperparameter search; latency is milliseconds per sample. Values are five-fold mean $\\pm$ sample SD.",
        "tab:s_timing",
        "lrrrrr",
        ["Surrogate", "Fit at $N=500$ (s)", "Fit at $N=10{,}000$ (s)", "Fit-time ratio", "Raw latency", "Projection latency"],
        rows,
    )


def write_full_hyperparameters() -> None:
    selected = json.loads((RUN_ROOT / "tuning" / "full_id_selected.json").read_text(encoding="utf-8"))
    rows = []
    for model_key in MODEL_ORDER:
        values = ", ".join(f"{key}={json.dumps(value, ensure_ascii=True)}" for key, value in selected[model_key].items())
        rows.append([LATEX_MODEL_LABELS[model_key], latex_escape(values)])
    longtable(
        OUTPUT / "table_s_full_hyperparameters.csv",
        "Selected settings for the final full-data surrogate refits used in out-of-distribution evaluation. Fixed implementation controls are included because they form part of the accepted fitting contract.",
        "tab:s_full_hyperparameters",
        "p{3.2cm}p{11.5cm}",
        ["Surrogate", "Selected full-data settings"],
        rows,
    )


def main() -> None:
    require_completed_run()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = RUN_ROOT / "metrics"
    id_components = pd.read_csv(metrics / "id_component_metrics.csv")
    id_composites = pd.read_csv(metrics / "id_composite_metrics.csv")
    id_summary = pd.read_csv(metrics / "id_fold_summary.csv")
    physical_summary = pd.read_csv(metrics / "id_physical_summary.csv")
    physical_fold = pd.read_csv(metrics / "id_physical_fold.csv")
    timing = pd.read_csv(metrics / "timing_fold_summary.csv")
    ood_components = pd.read_csv(metrics / "ood_component_metrics.csv")
    ood_composites = pd.read_csv(metrics / "ood_composite_metrics.csv")

    # Composite-normalized RMSE is derived only for the requested supplementary tables.
    # It uses the same no-leakage normalization convention as the component metrics:
    # outer-training targets for ID and all ID targets for OOD.
    id_dataset = pd.read_parquet(RUN_ROOT / "datasets" / "id.parquet")
    assignments = pd.read_parquet(RUN_ROOT / "splits" / "master_assignments.parquet")
    id_sigmas: list[dict[str, object]] = []
    for outer_fold in range(5):
        training_rows = assignments.loc[assignments["outer_fold"] != outer_fold, "row_index"].to_numpy(int)
        for composite in COMPOSITE_ORDER:
            id_sigmas.append({
                "outer_fold": outer_fold,
                "composite": composite,
                "composite_normalization_sd": float(id_dataset.iloc[training_rows][f"Out_{composite}"].std(ddof=0)),
            })
    id_composites = id_composites.merge(
        pd.DataFrame(id_sigmas), on=["outer_fold", "composite"], how="left", validate="many_to_one"
    )
    id_composites["nRMSE_composite"] = id_composites["RMSE"] / id_composites["composite_normalization_sd"]
    full_sigmas = {
        composite: float(id_dataset[f"Out_{composite}"].std(ddof=0)) for composite in COMPOSITE_ORDER
    }
    ood_composites["composite_normalization_sd"] = ood_composites["composite"].map(full_sigmas)
    ood_composites["nRMSE_composite"] = ood_composites["RMSE"] / ood_composites["composite_normalization_sd"]

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    plot_evaluation_workflow()
    plot_learning_curves(id_summary)
    plot_projection_change(id_summary)
    plot_violation_rates(physical_summary)
    plot_id_component_effects(id_components)
    plot_training_time(timing)
    plot_inference_latency(timing)
    plot_ood_component_effects(ood_components)
    write_accuracy_tables(id_components, id_composites, ood_components, ood_composites)
    write_backtracking_table(physical_fold)
    write_timing_table(timing)
    write_full_hyperparameters()
    print(f"Wrote manuscript-revision figures, tables, and source data to {OUTPUT}.")


if __name__ == "__main__":
    main()
