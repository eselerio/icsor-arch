"""Build a clearly labelled interim report from an in-progress benchmark bundle.

The script is intentionally read-only with respect to the run directory.  It only
uses model/size cells having all five outer folds for comparative summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
SAMPLE_SIZES = [500, 1450, 2400, 3350, 4300, 5250, 6200, 7150, 8100, 9050, 10000]
EXPECTED_FOLDS = set(range(5))
FILE_PATTERN = re.compile(r"^(?P<model>.+)_n(?P<size>\d+)_fold(?P<fold>\d+)\.json$")
COLORS = {
    "xgboost_regressor": "#0072B2",
    "lightgbm_regressor": "#009E73",
    "catboost_regressor": "#D55E00",
    "adaboost_regressor": "#CC79A7",
}


def load_records(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    metric_rows: list[dict] = []
    physical_rows: list[dict] = []
    timing_rows: list[dict] = []
    paths = sorted((run_root / "metrics" / "units").glob("*.json"))
    for path in paths:
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue
        metadata = {
            "model_key": match.group("model"),
            "sample_size": int(match.group("size")),
            "outer_fold": int(match.group("fold")),
        }
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        for row in record["metrics"]:
            metric_rows.append({**metadata, **row})
        physical_rows.append({**metadata, **record["physical_summary"]})
        timing_rows.append({**metadata, **record["timing"]})
    return pd.DataFrame(metric_rows), pd.DataFrame(physical_rows), pd.DataFrame(timing_rows), paths


def complete_cells(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = (
        metrics.groupby(["model_key", "sample_size"])["outer_fold"]
        .agg(lambda values: len(set(values)))
        .rename("fold_count")
        .reset_index()
    )
    complete = observed.query("fold_count == 5")[["model_key", "sample_size"]]
    usable = metrics.merge(complete, on=["model_key", "sample_size"], how="inner")
    return observed, usable


def summarize(metrics: pd.DataFrame, physical: pd.DataFrame, timing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_summary = (
        metrics.groupby(["model_key", "sample_size", "prediction_type"], sort=False)
        .agg(
            folds=("outer_fold", "nunique"),
            nRMSE_mean=("nRMSE", "mean"),
            nRMSE_sd=("nRMSE", "std"),
            nMAE_mean=("nMAE", "mean"),
            macro_R2_mean=("macro_R2", "mean"),
        )
        .reset_index()
    )
    keys = metric_summary[["model_key", "sample_size"]].drop_duplicates()
    physical_usable = physical.merge(keys, on=["model_key", "sample_size"], how="inner")
    physical_summary = (
        physical_usable.groupby(["model_key", "sample_size"], sort=False)
        .agg(
            folds=("outer_fold", "nunique"),
            raw_mass_violation_rate=("raw_mass_violation", "mean"),
            projected_mass_violation_rate=("projected_mass_violation", "mean"),
            raw_nonnegative_violation_rate=("raw_nonnegative_violation", "mean"),
            projected_nonnegative_violation_rate=("projected_nonnegative_violation", "mean"),
            raw_conservation_l2_mean=("raw_conservation_l2", "mean"),
            projected_conservation_l2_mean=("projected_conservation_l2", "mean"),
            standardized_displacement_l2_mean=("standardized_displacement_l2", "mean"),
            backtracking_active_mean=("backtracking_active", "mean"),
        )
        .reset_index()
    )
    timing_usable = timing.merge(keys, on=["model_key", "sample_size"], how="inner")
    timing_summary = (
        timing_usable.groupby(["model_key", "sample_size"], sort=False)
        .agg(
            folds=("outer_fold", "nunique"),
            setup_seconds_mean=("setup_seconds", "mean"),
            setup_seconds_sd=("setup_seconds", "std"),
            raw_latency_ms_mean=("raw_latency_ms_per_sample", "mean"),
            projection_latency_ms_mean=("projection_latency_ms_per_sample", "mean"),
            end_to_end_latency_ms_mean=("end_to_end_latency_ms_per_sample", "mean"),
        )
        .reset_index()
    )
    return metric_summary, physical_summary, timing_summary


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_completeness(observed: pd.DataFrame, output_dir: Path) -> None:
    matrix = (
        observed.pivot(index="model_key", columns="sample_size", values="fold_count")
        .reindex(index=MODEL_ORDER, columns=SAMPLE_SIZES)
        .fillna(0)
        .to_numpy(float)
    )
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    image = axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=5)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            axis.text(column, row, str(value), ha="center", va="center", color="white" if value >= 4 else "#222222", fontsize=7)
    axis.set_xticks(range(len(SAMPLE_SIZES)), [f"{value:,}" for value in SAMPLE_SIZES], rotation=45, ha="right")
    axis.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[key] for key in MODEL_ORDER])
    axis.set(xlabel="Nested sample total", title="Completed outer folds per model–size cell (5 required)")
    fig.colorbar(image, ax=axis, label="Completed folds", ticks=range(6), shrink=0.8)
    save(fig, output_dir, "01_run_completeness")


def plot_learning_curves(summary: pd.DataFrame, complete_models: list[str], output_dir: Path) -> None:
    raw = summary.query("prediction_type == 'raw'")
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for key in complete_models:
        rows = raw.query("model_key == @key").set_index("sample_size").reindex(SAMPLE_SIZES)
        x = np.asarray(SAMPLE_SIZES)
        mean = rows["nRMSE_mean"].to_numpy(float)
        sd = rows["nRMSE_sd"].to_numpy(float)
        axis.plot(x, mean, marker="o", markersize=3.5, linewidth=1.5, color=COLORS[key], label=MODEL_LABELS[key])
        axis.fill_between(x, mean - sd, mean + sd, color=COLORS[key], alpha=0.12, linewidth=0)
    axis.set(xlabel="Nested sample total", ylabel="Raw nRMSE", title="Five-fold in-distribution learning curves")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    save(fig, output_dir, "02_raw_learning_curves")


def plot_full_size(summary: pd.DataFrame, complete_models: list[str], output_dir: Path) -> None:
    rows = summary.query("sample_size == 10000 and model_key in @complete_models")
    fig, axis = plt.subplots(figsize=(7.4, 4.4))
    y = np.arange(len(complete_models))
    for offset, kind, color in [(-0.11, "raw", "#4C78A8"), (0.11, "projected", "#E45756")]:
        subset = rows.query("prediction_type == @kind").set_index("model_key").reindex(complete_models)
        axis.errorbar(
            subset["nRMSE_mean"], y + offset, xerr=subset["nRMSE_sd"], fmt="o", capsize=3,
            color=color, label=kind.capitalize(), markersize=5,
        )
    axis.set_yticks(y, [MODEL_LABELS[key] for key in complete_models])
    axis.invert_yaxis()
    axis.set(xlabel="nRMSE (mean ± fold SD)", title="Full-size ID accuracy (N = 10,000)")
    axis.grid(axis="x", alpha=0.2)
    axis.legend()
    save(fig, output_dir, "03_full_size_raw_projected")


def plot_projection_delta(summary: pd.DataFrame, complete_models: list[str], output_dir: Path) -> pd.DataFrame:
    pivot = summary.pivot(index=["model_key", "sample_size"], columns="prediction_type", values="nRMSE_mean").reset_index()
    pivot["delta_nRMSE"] = pivot["projected"] - pivot["raw"]
    matrix = np.vstack(
        [pivot.query("model_key == @key").set_index("sample_size").reindex(SAMPLE_SIZES)["delta_nRMSE"].to_numpy(float) for key in complete_models]
    )
    limit = float(np.nanmax(np.abs(matrix)))
    fig, axis = plt.subplots(figsize=(9.2, 3.1))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(SAMPLE_SIZES)), [f"{value:,}" for value in SAMPLE_SIZES], rotation=45, ha="right")
    axis.set_yticks(range(len(complete_models)), [MODEL_LABELS[key] for key in complete_models])
    axis.set(xlabel="Nested sample total", title="Projection effect on nRMSE (projected − raw)")
    fig.colorbar(image, ax=axis, label="ΔnRMSE; blue improves, red worsens", shrink=0.82)
    save(fig, output_dir, "04_projection_delta")
    return pivot


def plot_projection_trajectory(
    metrics: pd.DataFrame, summary: pd.DataFrame, complete_models: list[str], output_dir: Path
) -> pd.DataFrame:
    paired = metrics.pivot(
        index=["model_key", "sample_size", "outer_fold"], columns="prediction_type", values="nRMSE"
    ).reset_index()
    paired["delta_nRMSE"] = paired["projected"] - paired["raw"]
    paired_summary = (
        paired.groupby(["model_key", "sample_size"], sort=False)["delta_nRMSE"]
        .agg(delta_nRMSE_mean="mean", delta_nRMSE_sd="std")
        .reset_index()
    )
    means = summary.pivot(
        index=["model_key", "sample_size"], columns="prediction_type", values="nRMSE_mean"
    ).reset_index()
    means["relative_change_percent"] = 100.0 * (means["projected"] - means["raw"]) / means["raw"]
    trajectory = paired_summary.merge(
        means[["model_key", "sample_size", "raw", "projected", "relative_change_percent"]],
        on=["model_key", "sample_size"],
        how="left",
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    for key in complete_models:
        rows = trajectory.query("model_key == @key").set_index("sample_size").reindex(SAMPLE_SIZES)
        x = np.asarray(SAMPLE_SIZES)
        mean = rows["delta_nRMSE_mean"].to_numpy(float)
        sd = rows["delta_nRMSE_sd"].to_numpy(float)
        axes[0].plot(x, mean, marker="o", markersize=3.8, linewidth=1.5, color=COLORS[key], label=MODEL_LABELS[key])
        axes[0].fill_between(x, mean - sd, mean + sd, color=COLORS[key], alpha=0.10, linewidth=0)
        axes[1].plot(
            x, rows["relative_change_percent"].to_numpy(float), marker="o", markersize=3.8,
            linewidth=1.5, color=COLORS[key], label=MODEL_LABELS[key],
        )
    for axis in axes:
        axis.axhline(0, color="black", linewidth=1.0)
        axis.axhspan(axis.get_ylim()[0], 0, color="#009E73", alpha=0.05, zorder=-10)
        axis.grid(alpha=0.2)
    axes[0].set(ylabel="ΔnRMSE (projected − raw)", title="Absolute projection effect; band = SD of five paired-fold changes")
    axes[1].set(xlabel="Nested sample total", ylabel="Change relative to raw nRMSE (%)", title="Relative projection effect")
    axes[0].legend(ncol=2)
    axes[0].text(0.01, 0.04, "Improvement ↓", transform=axes[0].transAxes, color="#007A55", fontsize=8)
    axes[1].text(0.01, 0.04, "Improvement ↓", transform=axes[1].transAxes, color="#007A55", fontsize=8)
    fig.suptitle("How projection changes predictive performance across sample size", fontsize=12)
    fig.tight_layout()
    save(fig, output_dir, "06_projection_effect_by_sample_size")
    return trajectory


def plot_physics(physical: pd.DataFrame, complete_models: list[str], output_dir: Path) -> None:
    rows = physical.query("sample_size == 10000 and model_key in @complete_models").set_index("model_key").reindex(complete_models)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    y = np.arange(len(complete_models))
    raw = np.maximum(rows["raw_conservation_l2_mean"].to_numpy(float), 1e-16)
    projected = np.maximum(rows["projected_conservation_l2_mean"].to_numpy(float), 1e-16)
    axes[0].scatter(raw, y - 0.1, label="Raw", color="#4C78A8")
    axes[0].scatter(projected, y + 0.1, label="Projected", color="#E45756")
    axes[0].set_xscale("log")
    axes[0].set_yticks(y, [MODEL_LABELS[key] for key in complete_models])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Mean conservation L2 residual", title="Conservation residual")
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].legend()
    width = 0.36
    axes[1].barh(y - width / 2, rows["raw_mass_violation_rate"] * 100, height=width, label="Raw", color="#4C78A8")
    axes[1].barh(y + width / 2, rows["projected_mass_violation_rate"] * 100, height=width, label="Projected", color="#E45756")
    axes[1].set_yticks(y, [MODEL_LABELS[key] for key in complete_models])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Fold-mean sample violation rate (%)", title="Mass-conservation violations")
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].legend()
    fig.suptitle("Full-size physical admissibility (N = 10,000)")
    fig.tight_layout()
    save(fig, output_dir, "05_full_size_physics")


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return "—"
    return f"{value:.{digits}f}"


def write_report(
    output_dir: Path,
    run_root: Path,
    manifest: dict,
    captured_at: datetime,
    observed: pd.DataFrame,
    summary: pd.DataFrame,
    physical: pd.DataFrame,
    timing: pd.DataFrame,
    delta: pd.DataFrame,
    metric_paths: list[Path],
    complete_models: list[str],
) -> None:
    full = summary.query("sample_size == 10000 and model_key in @complete_models")
    full_pivot = full.pivot(index="model_key", columns="prediction_type", values=["nRMSE_mean", "nRMSE_sd", "nMAE_mean", "macro_R2_mean"])
    phys = physical.query("sample_size == 10000 and model_key in @complete_models").set_index("model_key")
    times = timing.query("sample_size == 10000 and model_key in @complete_models").set_index("model_key")
    delta_full = delta.query("sample_size == 10000").set_index("model_key")
    total_expected = len(MODEL_ORDER) * len(SAMPLE_SIZES) * 5
    completed_fold_files = int(observed["fold_count"].sum())
    complete_cells_count = int((observed["fold_count"] == 5).sum())
    negative_cells = int((delta["delta_nRMSE"] < 0).sum())
    positive_cells = int((delta["delta_nRMSE"] > 0).sum())
    lines = [
        "# Interim benchmark results",
        "",
        "> **Partial, non-final evidence.** The source run was still marked `running` when captured. "
        "Only model–size cells with all five outer folds are used for means, standard deviations, and comparative claims.",
        "",
        f"- Run: `{manifest.get('run_id', run_root.name)}`",
        f"- Captured: {captured_at.astimezone().isoformat(timespec='seconds')}",
        f"- Manifest updated: {manifest.get('updated_utc', 'unknown')}",
        f"- Fold metric files observed: {len(metric_paths)} of {total_expected} planned ({100 * len(metric_paths) / total_expected:.1f}%)",
        f"- Complete five-fold model–size cells: {complete_cells_count} of {len(MODEL_ORDER) * len(SAMPLE_SIZES)}",
        f"- Fully comparable models across all 11 sizes: {', '.join(MODEL_LABELS[key] for key in complete_models)}",
        "",
        "![Run completeness](01_run_completeness.png)",
        "",
        "## Main interim findings",
        "",
        "The four completed models improve as the training pool grows. Their five-fold raw nRMSE curves are shown below; shaded regions are fold SD, not confidence intervals.",
        "",
        "![Raw learning curves](02_raw_learning_curves.png)",
        "",
        "At the full in-distribution size, the available results are:",
        "",
        "| Model | Raw nRMSE | Projected nRMSE | ΔnRMSE | Raw nMAE | Raw macro R² | Setup (s) | Raw latency (ms/sample) | Projection latency (ms/sample) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in complete_models:
        raw_mean = full_pivot.loc[key, ("nRMSE_mean", "raw")]
        raw_sd = full_pivot.loc[key, ("nRMSE_sd", "raw")]
        projected_mean = full_pivot.loc[key, ("nRMSE_mean", "projected")]
        projected_sd = full_pivot.loc[key, ("nRMSE_sd", "projected")]
        lines.append(
            f"| {MODEL_LABELS[key]} | {fmt(raw_mean)} ± {fmt(raw_sd)} | {fmt(projected_mean)} ± {fmt(projected_sd)} | "
            f"{fmt(delta_full.loc[key, 'delta_nRMSE'])} | {fmt(full_pivot.loc[key, ('nMAE_mean', 'raw')])} | "
            f"{fmt(full_pivot.loc[key, ('macro_R2_mean', 'raw')])} | {fmt(times.loc[key, 'setup_seconds_mean'], 1)} | "
            f"{fmt(times.loc[key, 'raw_latency_ms_mean'], 4)} | {fmt(times.loc[key, 'projection_latency_ms_mean'], 4)} |"
        )
    lines.extend(
        [
            "",
            "![Full-size accuracy](03_full_size_raw_projected.png)",
            "",
            f"Across the {len(delta)} complete model–size cells, projection lowers mean nRMSE in {negative_cells} and raises it in {positive_cells}. "
            "This confirms that physical enforcement and predictive accuracy are separate outcomes: the projection guarantee does not imply an accuracy gain.",
            "",
            "![Projection delta heatmap](04_projection_delta.png)",
            "",
            "The corresponding line trajectories below show both the absolute paired-fold change and the percentage change relative to raw nRMSE. Values below zero indicate improvement.",
            "",
            "![Projection effect by sample size](06_projection_effect_by_sample_size.png)",
            "",
            "## Physical enforcement",
            "",
            "At N = 10,000, every available raw fold reports mass-conservation violations, whereas projected violation rates are zero. "
            "The projected mean conservation residuals are near floating-point precision. Full-size non-negativity violation rates are tabulated separately below.",
            "",
            "| Model | Raw mass violation | Projected mass violation | Raw nonnegative violation | Projected nonnegative violation | Projected conservation L2 | Mean standardized displacement L2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in complete_models:
        row = phys.loc[key]
        lines.append(
            f"| {MODEL_LABELS[key]} | {100 * row.raw_mass_violation_rate:.1f}% | {100 * row.projected_mass_violation_rate:.1f}% | "
            f"{100 * row.raw_nonnegative_violation_rate:.2f}% | {100 * row.projected_nonnegative_violation_rate:.2f}% | "
            f"{row.projected_conservation_l2_mean:.2e} | {fmt(row.standardized_displacement_l2_mean)} |"
        )
    lines.extend(
        [
            "",
            "![Full-size physical diagnostics](05_full_size_physics.png)",
            "",
            "## Interpretation boundary",
            "",
            "Random Forest currently has one fold at each size. Those values are preserved in the source snapshot but are not mixed into the five-fold comparisons. "
            "Extra Trees, SVR, k-NN, PLS, Multi-task Elastic Net, Multi-task Lasso, MLP, and TabNet have no scored folds in this capture. "
            "OOD analyses and final fitted-model artifacts are also unavailable, so no extrapolation conclusion or final 13-model ranking is warranted.",
            "",
            "These figures are exploratory interim diagnostics and should not replace the manuscript placeholders until the run writes its completion sentinel and passes the terminal validation assertions.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now(timezone.utc)
    with (run_root / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    metrics, physical, timing, paths = load_records(run_root)
    observed, usable_metrics = complete_cells(metrics)
    usable_keys = usable_metrics[["model_key", "sample_size"]].drop_duplicates()
    usable_physical = physical.merge(usable_keys, on=["model_key", "sample_size"], how="inner")
    usable_timing = timing.merge(usable_keys, on=["model_key", "sample_size"], how="inner")
    summary, physical_summary, timing_summary = summarize(usable_metrics, usable_physical, usable_timing)
    complete_models = [
        key for key in MODEL_ORDER
        if set(observed.query("model_key == @key and fold_count == 5")["sample_size"]) == set(SAMPLE_SIZES)
    ]
    configure_style()
    plot_completeness(observed, output_dir)
    plot_learning_curves(summary, complete_models, output_dir)
    plot_full_size(summary, complete_models, output_dir)
    delta = plot_projection_delta(summary, complete_models, output_dir)
    trajectory = plot_projection_trajectory(usable_metrics, summary, complete_models, output_dir)
    plot_physics(physical_summary, complete_models, output_dir)
    observed.to_csv(output_dir / "completeness.csv", index=False)
    summary.to_csv(output_dir / "id_metric_summary_complete_cells.csv", index=False)
    physical_summary.to_csv(output_dir / "id_physical_summary_complete_cells.csv", index=False)
    timing_summary.to_csv(output_dir / "id_timing_summary_complete_cells.csv", index=False)
    delta.to_csv(output_dir / "projection_delta_complete_cells.csv", index=False)
    trajectory.to_csv(output_dir / "projection_effect_by_sample_size.csv", index=False)
    write_report(
        output_dir, run_root, manifest, captured_at, observed, summary, physical_summary,
        timing_summary, delta, paths, complete_models,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
