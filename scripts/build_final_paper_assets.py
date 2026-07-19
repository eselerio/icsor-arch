"""Validate a completed benchmark and build manuscript-ready result assets.

This script is intentionally separate from ``main.ipynb``.  The notebook owns the
immutable scientific run; this utility reads that completed bundle and writes only
derived paper assets.  By default it refuses smoke, partial, or hash-inconsistent
runs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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
MODEL_CATEGORIES = {
    "xgboost_regressor": "Boosting",
    "lightgbm_regressor": "Boosting",
    "catboost_regressor": "Boosting",
    "adaboost_regressor": "Boosting",
    "random_forest_regressor": "Randomized tree ensembles",
    "extra_trees_regressor": "Randomized tree ensembles",
    "svr_regressor": "Kernel",
    "knn_regressor": "Instance based",
    "pls_regressor": "Interpretable models",
    "multitask_elastic_net_regressor": "Interpretable models",
    "multitask_lasso_regressor": "Interpretable models",
    "ann_deep_regressor": "Neural networks",
    "tabnet_regressor": "Neural networks",
}
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
    "xgboost_regressor": "o",
    "lightgbm_regressor": "s",
    "catboost_regressor": "^",
    "adaboost_regressor": "v",
    "random_forest_regressor": "D",
    "extra_trees_regressor": "d",
    "svr_regressor": "P",
    "knn_regressor": ">",
    "pls_regressor": "<",
    "multitask_elastic_net_regressor": "h",
    "multitask_lasso_regressor": "+",
    "ann_deep_regressor": "p",
    "tabnet_regressor": "*",
}
STATE_COLUMNS = [
    "S_O", "S_F", "S_A", "S_NH4", "S_NO2", "S_NO3", "S_N2", "S_PO4",
    "S_I", "S_ALK", "X_I", "X_S", "X_H", "X_PAO", "X_PP", "X_PHA",
    "X_AOB", "X_NOB", "X_MeP", "X_MeOH",
]
FULL_SAMPLE_SIZES = [500, 1450, 2400, 3350, 4300, 5250, 6200, 7150, 8100, 9050, 10000]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = json.loads(json.dumps(value))
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_bundle(run_root: Path, *, allow_ineligible: bool, verify_inventory: bool) -> dict[str, Any]:
    completed_path = run_root / "COMPLETED.json"
    manifest_path = run_root / "manifest.json"
    manifest_hash_path = run_root / "manifest.sha256"
    require(completed_path.is_file(), f"Missing completion sentinel: {completed_path}")
    require(manifest_path.is_file(), f"Missing manifest: {manifest_path}")
    require(manifest_hash_path.is_file(), f"Missing manifest hash: {manifest_hash_path}")

    completed = load_json(completed_path)
    manifest = load_json(manifest_path)
    require(manifest.get("status") == "complete", "Manifest status is not complete.")
    require(completed.get("run_id") == manifest.get("run_id") == run_root.name, "Run IDs do not agree.")
    require(completed.get("manifest_sha256") == sha256_file(manifest_path), "COMPLETED.json manifest hash mismatch.")
    declared = manifest_hash_path.read_text(encoding="utf-8").split()[0]
    require(declared == sha256_file(manifest_path), "manifest.sha256 mismatch.")
    if not allow_ineligible:
        require(completed.get("profile") == "full", "Only the full profile may populate the paper.")
        require(completed.get("article_eligible") is True, "Run is not article eligible.")
        require(manifest.get("profile") == "full", "Manifest is not a full-profile run.")
        require(manifest.get("article_eligible") is True, "Manifest is not article eligible.")

    if verify_inventory:
        inventory = manifest.get("artifact_inventory", {})
        for relative, record in inventory.items():
            path = run_root / relative
            require(path.is_file(), f"Manifest artifact is absent: {relative}")
            require(path.stat().st_size == int(record["bytes"]), f"Size mismatch: {relative}")
            require(sha256_file(path) == record["sha256"], f"SHA-256 mismatch: {relative}")
        registered = set(inventory)
        actual = {
            path.relative_to(run_root).as_posix()
            for path in run_root.rglob("*")
            if path.is_file()
        }
        require(
            actual - registered == {"COMPLETED.json", "manifest.json", "manifest.sha256"},
            f"Unregistered run files: {sorted(actual - registered)}",
        )
    return manifest


def verify_source_contract(run_root: Path, manifest: dict[str, Any], source_root: Path) -> dict[str, str]:
    contract = manifest.get("contract", {})
    observed: dict[str, str] = {}
    for field, path in {
        "notebook_sha256": source_root / "main.ipynb",
        "dependency_lock_sha256": source_root / "uv.lock",
    }.items():
        require(path.is_file(), f"Current contract source is missing: {path}")
        observed[field] = sha256_file(path)
        require(observed[field] == contract.get(field), f"Current {path.name} differs from the accepted run contract.")

    current_config = load_json(source_root / "config" / "params.json")
    profile_override = json.loads(json.dumps(current_config["run"]["profiles"].get("full", {})))
    profile_run_id = profile_override.pop("run_id", None)
    active = deep_merge(current_config, profile_override)
    if profile_run_id is not None:
        active["run"]["run_id"] = profile_run_id
    active["run"]["profile"] = "full"
    resolved = load_json(run_root / "inputs" / "params.resolved.json")
    require(active == resolved, "Current full-profile configuration differs from the accepted resolved configuration.")
    observed["config_sha256"] = sha256_json(active)
    require(observed["config_sha256"] == contract.get("config_sha256"), "Resolved configuration hash differs from the run contract.")

    current_paths = load_json(source_root / "config" / "paths.json")
    observed["paths_sha256"] = sha256_json(current_paths)
    require(observed["paths_sha256"] == contract.get("paths_sha256"), "Paths configuration hash differs from the run contract.")
    workbook_snapshot = run_root / "inputs" / "workbook.snapshot.xlsx"
    observed["workbook_sha256"] = sha256_file(workbook_snapshot)
    require(observed["workbook_sha256"] == contract.get("workbook_sha256"), "Workbook snapshot differs from the run contract.")
    current_workbook = (source_root / str(current_paths["workbook"])).resolve()
    require(current_workbook.is_file(), f"Configured workbook is missing: {current_workbook}")
    require(sha256_file(current_workbook) == observed["workbook_sha256"], "Current configured workbook differs from the accepted snapshot.")
    return observed


def validate_scientific_contract(
    run_root: Path,
    manifest: dict[str, Any],
    resolved: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    *,
    strict_full: bool,
) -> dict[str, Any]:
    stages = manifest.get("stages", {})
    expected_stages = {
        "workbook_and_matrices", "mechanistic_tests", "id_dataset", "projection_tests",
        "split_tests", "id_benchmark", "full_id_selection", "full_id_refits",
        "ood_benchmark", "publication_assets",
    }
    require(set(stages) == expected_stages, f"Unexpected manifest stage set: {sorted(stages)}")
    require(all(record.get("status") == "complete" for record in stages.values()), "At least one manifest stage is incomplete.")
    require(int(resolved["simulation"]["sampling"]["seed"]) == 42, "The accepted sampling seed is not 42.")
    require(float(resolved["projection"]["tau"]) == 1e-10, "The physical-violation tolerance is not 1e-10.")
    if strict_full:
        nested = resolved["evaluation"]["nested_cv"]
        timing = resolved["evaluation"]["timing"]
        regimes = resolved["simulation"]["ood"]["regimes"]
        require(int(resolved["simulation"]["sampling"]["id_samples"]) == 10_000, "Full ID target is not 10,000.")
        require(list(nested["sample_sizes"]) == FULL_SAMPLE_SIZES, "Resolved full sample-size grid differs.")
        require(int(nested["outer_folds"]) == 5 and int(nested["inner_folds"]) == 4, "Resolved CV is not 5x4.")
        require(int(nested["n_trials"]) == 100, "Resolved search budget is not 100 trials.")
        require(
            (int(timing["batch_size"]), int(timing["warmup_runs"]), int(timing["measured_runs"])) == (512, 5, 30),
            "Resolved timing protocol is not 512 rows, five warm-ups, and 30 measured calls.",
        )
        require(all(int(regimes[name]["n_samples"]) == 600 for name in ("mild", "severe")), "OOD target is not 600 per stratum.")

    maximum_repeat_difference = float(
        max(
            tables["timing_fold"]["maximum_repeated_prediction_difference"].max(),
            tables["ood_timing"]["maximum_repeated_prediction_difference"].max(),
        )
    )
    require(maximum_repeat_difference <= 1e-12, f"Repeated predictions differ by {maximum_repeat_difference:.6g}.")

    model_paths = sorted((run_root / "models").glob("*.joblib"))
    require({path.stem for path in model_paths} == set(MODEL_ORDER), "Final fitted-model roster is incomplete.")
    coefficient_keys = ["pls_regressor", "multitask_elastic_net_regressor", "multitask_lasso_regressor"]
    for key in coefficient_keys:
        coefficients = pd.read_csv(run_root / "models" / "interpretability" / f"{key}_standardized_coefficients.csv")
        require(len(coefficients) == 22, f"{key} coefficient export does not contain 22 inputs.")
        require(coefficients["feature"].nunique() == 22, f"{key} coefficient features are not unique.")
        require(set(STATE_COLUMNS).issubset(coefficients.columns), f"{key} lacks one or more 20-response coefficients.")
        values = coefficients[STATE_COLUMNS].to_numpy(float)
        require(np.isfinite(values).all(), f"{key} contains a non-finite coefficient.")
        norms = np.linalg.norm(values, axis=1)
        require(np.allclose(norms, coefficients["row_l2_norm"].to_numpy(float), rtol=1e-12, atol=1e-14), f"{key} row norms do not reproduce.")
        require(np.array_equal(norms > 1e-12, coefficients["active_at_1e-12"].to_numpy(bool)), f"{key} activity flags do not reproduce.")
        require((coefficients["model_category"] == "Interpretable models").all(), f"{key} has the wrong reporting category.")

    selected_paths = sorted((run_root / "tuning").glob("*/*.selected.json"))
    trial_paths = sorted((run_root / "tuning").glob("*/*.trials.csv"))
    study_timing_paths = sorted((run_root / "tuning").glob("*/*.timing.json"))
    expected_studies = len(MODEL_ORDER) * (6 if strict_full else len(tables["id_fold"]["outer_fold"].unique()) + 1)
    require(len(selected_paths) == expected_studies, f"Expected {expected_studies} selected configurations, found {len(selected_paths)}.")
    require(len(trial_paths) == expected_studies, f"Expected {expected_studies} trial exports, found {len(trial_paths)}.")
    require(len(study_timing_paths) == expected_studies, f"Expected {expected_studies} study-timing records, found {len(study_timing_paths)}.")
    fold_ids = sorted(tables["id_fold"]["outer_fold"].unique().tolist())
    study_labels = [f"outer_{fold}" for fold in fold_ids] + ["full_id"]
    expected_selected = {f"tuning/{key}/{label}.selected.json" for key in MODEL_ORDER for label in study_labels}
    expected_trials_grid = {f"tuning/{key}/{label}.trials.csv" for key in MODEL_ORDER for label in study_labels}
    expected_timing_grid = {f"tuning/{key}/{label}.timing.json" for key in MODEL_ORDER for label in study_labels}
    require({path.relative_to(run_root).as_posix() for path in selected_paths} == expected_selected, "Selected-configuration study grid is incomplete.")
    require({path.relative_to(run_root).as_posix() for path in trial_paths} == expected_trials_grid, "Trial-export study grid is incomplete.")
    require({path.relative_to(run_root).as_posix() for path in study_timing_paths} == expected_timing_grid, "Study-timing grid is incomplete.")
    expected_trials = 100 if strict_full else int(resolved["evaluation"]["nested_cv"]["n_trials"])
    for path in trial_paths:
        trials = pd.read_csv(path)
        require(len(trials) == expected_trials, f"{path.relative_to(run_root)} has {len(trials)} rather than {expected_trials} trials.")
        require((trials["state"] == "COMPLETE").all(), f"{path.relative_to(run_root)} contains a non-COMPLETE trial.")
        require(set(trials["number"].astype(int)) == set(range(expected_trials)), f"{path.relative_to(run_root)} trial numbers are incomplete.")
        require(np.isfinite(trials["value"].to_numpy(float)).all(), f"{path.relative_to(run_root)} contains a non-finite objective.")
        timing_path = path.with_name(path.name.replace(".trials.csv", ".timing.json"))
        study_timing = load_json(timing_path)
        require(int(study_timing.get("completed_trials", -1)) == expected_trials, f"{timing_path.relative_to(run_root)} completed-trial count disagrees with the trial export.")
        search_seconds = float(study_timing.get("search_seconds", float("nan")))
        require(np.isfinite(search_seconds) and search_seconds >= 0.0, f"{timing_path.relative_to(run_root)} has an invalid search duration.")
        selected_path = path.with_name(path.name.replace(".trials.csv", ".selected.json"))
        selected = load_json(selected_path)
        best = trials.loc[trials["value"].astype(float).idxmin()]
        for column in [name for name in trials.columns if name.startswith("params_")]:
            parameter = column.removeprefix("params_")
            require(parameter in selected, f"Selected configuration omits tuned parameter {parameter} in {selected_path.name}.")
            observed, expected_value = best[column], selected[parameter]
            if expected_value is None:
                require(pd.isna(observed) or str(observed).lower() == "none", f"Best-trial parameter {parameter} does not match the selected null value.")
            elif isinstance(expected_value, bool):
                require(str(observed).lower() == str(expected_value).lower(), f"Best-trial Boolean {parameter} does not match selection.")
            elif isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
                require(np.isclose(float(observed), float(expected_value), rtol=1e-12, atol=1e-15), f"Best-trial numeric parameter {parameter} does not match selection.")
            else:
                require(str(observed) == str(expected_value), f"Best-trial categorical parameter {parameter} does not match selection.")
    return {
        "completed_stages": sorted(stages),
        "maximum_repeated_prediction_difference": maximum_repeat_difference,
        "selected_configuration_count": len(selected_paths),
        "trial_export_count": len(trial_paths),
        "study_timing_count": len(study_timing_paths),
        "trials_per_study": expected_trials,
        "fitted_model_count": len(model_paths),
        "coefficient_artifact_count": len(coefficient_keys),
    }


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    categorical_orders = {
        "model_key": MODEL_ORDER,
        "prediction_type": ["raw", "projected"],
        "regime": ["mild", "severe"],
        "component": STATE_COLUMNS,
        "composite": ["COD", "TN", "TP", "TSS"],
    }
    for column, categories in categorical_orders.items():
        if column in result:
            result[column] = pd.Categorical(result[column], categories, ordered=True)
    sort_order = [
        "model_key", "sample_size", "outer_fold", "regime", "prediction_type",
        "component", "composite",
    ]
    return result.sort_values([column for column in sort_order if column in result.columns])


def assert_exact_grid(frame: pd.DataFrame, keys: list[str], dimensions: list[Iterable[Any]], name: str) -> None:
    observed = set(frame[keys].itertuples(index=False, name=None))
    expected = set(itertools.product(*dimensions))
    require(len(frame) == len(observed), f"{name} has duplicate keys.")
    require(observed == expected, f"{name} does not contain the exact expected key grid.")


def reconcile_fold_summary(
    fold_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    groups: list[str],
    values: list[str],
    name: str,
) -> float:
    recomputed = fold_frame.groupby(groups, sort=False)[values].agg(["mean", "std"])
    recomputed.columns = [f"{value}_{statistic}" for value, statistic in recomputed.columns]
    observed = summary_frame.set_index(groups)
    require(observed.index.is_unique, f"{name} summary keys are not unique.")
    recomputed = recomputed.sort_index()
    observed = observed.sort_index()
    require(recomputed.index.equals(observed.index), f"{name} summary keys disagree with fold rows.")
    maximum = 0.0
    for column in recomputed.columns:
        calculated = recomputed[column].to_numpy(float)
        stored = observed[column].to_numpy(float)
        maximum = max(maximum, float(np.max(np.abs(calculated - stored))))
        require(np.allclose(calculated, stored, rtol=1e-11, atol=1e-13), f"{name} column {column} does not reconcile.")
    counts = fold_frame.groupby(groups, sort=False).size().sort_index().to_numpy(int)
    require(np.array_equal(counts, observed["fold_count"].to_numpy(int)), f"{name} fold counts do not reconcile.")
    return maximum


def load_and_validate_tables(run_root: Path, *, strict_full: bool) -> dict[str, pd.DataFrame]:
    paths = {
        "id_fold": run_root / "metrics" / "id_fold_metrics.csv",
        "id_summary": run_root / "metrics" / "id_fold_summary.csv",
        "timing_fold": run_root / "timing" / "id_fold_timing.csv",
        "timing_summary": run_root / "metrics" / "timing_fold_summary.csv",
        "physical_fold": run_root / "metrics" / "id_physical_fold.csv",
        "physical_summary": run_root / "metrics" / "id_physical_summary.csv",
        "component": run_root / "metrics" / "id_component_metrics.csv",
        "composite": run_root / "metrics" / "id_composite_metrics.csv",
        "ood": run_root / "metrics" / "ood_metrics.csv",
        "ood_physical": run_root / "metrics" / "ood_physical_summary.csv",
        "ood_component": run_root / "metrics" / "ood_component_metrics.csv",
        "ood_composite": run_root / "metrics" / "ood_composite_metrics.csv",
        "ood_timing": run_root / "timing" / "ood_timing.csv",
    }
    for name, path in paths.items():
        require(path.is_file(), f"Missing required {name} table: {path}")
    tables = {name: pd.read_csv(path) for name, path in paths.items()}

    model_keys = set(tables["id_fold"]["model_key"])
    require(model_keys == set(MODEL_ORDER), f"Unexpected model roster: {sorted(model_keys)}")
    require(set(tables["ood"]["model_key"]) == set(MODEL_ORDER), "OOD model roster mismatch.")
    sizes = sorted(tables["id_fold"]["sample_size"].unique().tolist())
    folds = sorted(tables["id_fold"]["outer_fold"].unique().tolist())
    if strict_full:
        require(sizes == FULL_SAMPLE_SIZES, f"Unexpected full-profile sample sizes: {sizes}")
        require(folds == list(range(5)), f"Unexpected outer folds: {folds}")
    n_models, n_sizes, n_folds = len(MODEL_ORDER), len(sizes), len(folds)
    expected = {
        "id_fold": n_models * n_sizes * n_folds * 2,
        "id_summary": n_models * n_sizes * 2,
        "timing_fold": n_models * n_sizes * n_folds,
        "timing_summary": n_models * n_sizes,
        "physical_fold": n_models * n_sizes * n_folds * 2,
        "physical_summary": n_models * n_sizes * 2,
        "component": n_models * n_sizes * n_folds * 2 * len(STATE_COLUMNS),
        "composite": n_models * n_sizes * n_folds * 2 * 4,
        "ood": n_models * 2 * 2,
        "ood_physical": n_models * 2 * 2,
        "ood_component": n_models * 2 * 2 * len(STATE_COLUMNS),
        "ood_composite": n_models * 2 * 2 * 4,
        "ood_timing": n_models * 2,
    }
    for name, count in expected.items():
        require(len(tables[name]) == count, f"{name}: expected {count} rows, found {len(tables[name])}.")

    models, predictions = MODEL_ORDER, ["raw", "projected"]
    regimes, components, composites = ["mild", "severe"], STATE_COLUMNS, ["COD", "TN", "TP", "TSS"]
    assert_exact_grid(tables["id_fold"], ["model_key", "sample_size", "outer_fold", "prediction_type"], [models, sizes, folds, predictions], "id_fold")
    assert_exact_grid(tables["id_summary"], ["model_key", "sample_size", "prediction_type"], [models, sizes, predictions], "id_summary")
    assert_exact_grid(tables["timing_fold"], ["model_key", "sample_size", "outer_fold"], [models, sizes, folds], "timing_fold")
    assert_exact_grid(tables["timing_summary"], ["model_key", "sample_size"], [models, sizes], "timing_summary")
    assert_exact_grid(tables["physical_fold"], ["model_key", "sample_size", "outer_fold", "prediction_type"], [models, sizes, folds, predictions], "physical_fold")
    assert_exact_grid(tables["physical_summary"], ["model_key", "sample_size", "prediction_type"], [models, sizes, predictions], "physical_summary")
    assert_exact_grid(tables["component"], ["model_key", "sample_size", "outer_fold", "prediction_type", "component"], [models, sizes, folds, predictions, components], "component")
    assert_exact_grid(tables["composite"], ["model_key", "sample_size", "outer_fold", "prediction_type", "composite"], [models, sizes, folds, predictions, composites], "composite")
    assert_exact_grid(tables["ood"], ["model_key", "regime", "prediction_type"], [models, regimes, predictions], "ood")
    assert_exact_grid(tables["ood_physical"], ["model_key", "regime", "prediction_type"], [models, regimes, predictions], "ood_physical")
    assert_exact_grid(tables["ood_component"], ["model_key", "regime", "prediction_type", "component"], [models, regimes, predictions, components], "ood_component")
    assert_exact_grid(tables["ood_composite"], ["model_key", "regime", "prediction_type", "composite"], [models, regimes, predictions, composites], "ood_composite")
    assert_exact_grid(tables["ood_timing"], ["model_key", "regime"], [models, regimes], "ood_timing")

    for name, frame in tables.items():
        if "model_key" not in frame:
            continue
        labels = frame[["model_key", "model", "model_category"]].drop_duplicates()
        require(len(labels) == len(MODEL_ORDER), f"{name} model metadata are not one-to-one.")
        for row in labels.itertuples(index=False):
            require(row.model == MODEL_LABELS[row.model_key], f"{name} has the wrong label for {row.model_key}.")
            require(row.model_category == MODEL_CATEGORIES[row.model_key], f"{name} has the wrong category for {row.model_key}.")

    finite_columns = {
        "id_fold": ["nMSE", "nRMSE", "nMAE", "macro_R2"],
        "timing_fold": ["setup_seconds", "search_seconds", "raw_latency_ms_per_sample", "projection_latency_ms_per_sample", "end_to_end_latency_ms_per_sample", "maximum_repeated_prediction_difference"],
        "physical_fold": ["conservation_l2_mean", "conservation_l2_std", "conservation_l2_median", "conservation_l2_q95", "conservation_l2_max", "negative_standardized_l1_mean", "negative_standardized_l1_q95", "mass_violation_rate", "nonnegative_violation_rate", "standardized_displacement_l2_mean", "backtracking_rate", "backtracking_alpha_mean", "backtracking_alpha_min"],
        "ood": ["nMSE", "nRMSE", "nMAE", "macro_R2", "mass_violation_rate", "nonnegative_violation_rate"],
        "ood_physical": ["conservation_l2_mean", "conservation_l2_std", "conservation_l2_median", "conservation_l2_q95", "conservation_l2_max", "mass_violation_rate", "nonnegative_violation_rate", "standardized_displacement_l2_mean", "backtracking_rate", "backtracking_alpha_mean", "backtracking_alpha_min"],
    }
    for name, columns in finite_columns.items():
        require(np.isfinite(tables[name][columns].to_numpy(float)).all(), f"{name} contains a non-finite required value.")

    reconcile_fold_summary(
        tables["id_fold"], tables["id_summary"],
        ["model", "model_key", "model_category", "sample_size", "prediction_type"],
        ["nMSE", "nRMSE", "nMAE", "macro_R2"], "ID metric",
    )
    reconcile_fold_summary(
        tables["timing_fold"], tables["timing_summary"],
        ["model", "model_key", "model_category", "sample_size"],
        ["setup_seconds", "search_seconds", "raw_latency_ms_per_sample", "projection_latency_ms_per_sample", "end_to_end_latency_ms_per_sample", "maximum_repeated_prediction_difference"],
        "timing",
    )
    reconcile_fold_summary(
        tables["physical_fold"], tables["physical_summary"],
        ["model", "model_key", "model_category", "sample_size", "prediction_type"],
        ["conservation_l2_mean", "conservation_l2_std", "conservation_l2_median", "conservation_l2_q95", "conservation_l2_max", "negative_standardized_l1_mean", "negative_standardized_l1_q95", "mass_violation_rate", "nonnegative_violation_rate", "standardized_displacement_l2_mean", "backtracking_rate", "backtracking_alpha_mean", "backtracking_alpha_min"],
        "physical diagnostic",
    )
    return tables


def normalized_metrics(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    standardized = (y_pred - y_true) / sigma
    nmse = float(np.mean(np.square(standardized)))
    residual_sum = np.sum(np.square(y_true - y_pred), axis=0)
    total_sum = np.sum(np.square(y_true - np.mean(y_true, axis=0)), axis=0)
    require(np.all(total_sum > 0), "A validation component is constant, so macro R2 is undefined.")
    return {
        "nMSE": nmse,
        "nRMSE": math.sqrt(nmse),
        "nMAE": float(np.mean(np.abs(standardized))),
        "macro_R2": float(np.mean(1.0 - residual_sum / total_sum)),
    }


def prediction_contract(
    run_root: Path,
    tau: float,
    tables: dict[str, pd.DataFrame],
    *,
    strict_full: bool,
) -> dict[str, Any]:
    id_paths = sorted((run_root / "predictions" / "id").glob("*.parquet"))
    ood_paths = sorted((run_root / "predictions" / "ood").glob("*.parquet"))
    if strict_full:
        require(len(id_paths) == 13 * 11 * 5, f"Expected 715 ID prediction files, found {len(id_paths)}.")
        require(len(ood_paths) == 13 * 2, f"Expected 26 OOD prediction files, found {len(ood_paths)}.")
    total_instances = 0
    id_instances = 0
    max_conservation = -math.inf
    min_component = math.inf
    mass_violations = 0
    nonnegative_violations = 0
    maximum_metric_difference = 0.0
    maximum_diagnostic_difference = 0.0
    maximum_component_difference = 0.0
    maximum_composite_difference = 0.0
    operators = np.load(run_root / "matrices" / "operators.npz")
    invariant = np.asarray(operators["invariant"], dtype=float)
    composition = np.asarray(operators["composition"], dtype=float)
    id_data = pd.read_parquet(
        run_root / "datasets" / "id.parquet",
        columns=[
            "sample_id",
            *[f"In_{name}" for name in STATE_COLUMNS],
            *[f"Out_{name}" for name in STATE_COLUMNS],
        ],
    )
    id_data["sample_id"] = id_data["sample_id"].astype(str)
    id_by_sample = id_data.set_index("sample_id")
    require(id_by_sample.index.is_unique, "ID dataset sample IDs are not unique.")
    splits = pd.read_parquet(run_root / "splits" / "master_assignments.parquet")
    splits["sample_id"] = splits["sample_id"].astype(str)
    id_metric_index = tables["id_fold"].set_index(
        ["model_key", "sample_size", "outer_fold", "prediction_type"]
    )
    require(id_metric_index.index.is_unique, "ID metric keys are not unique.")
    ood_metric_index = tables["ood"].set_index(
        ["model_key", "regime", "prediction_type"]
    )
    require(ood_metric_index.index.is_unique, "OOD metric keys are not unique.")
    id_component_index = tables["component"].set_index(
        ["model_key", "sample_size", "outer_fold", "prediction_type", "component"]
    )
    require(id_component_index.index.is_unique, "ID component metric keys are not unique.")
    id_composite_index = tables["composite"].set_index(
        ["model_key", "sample_size", "outer_fold", "prediction_type", "composite"]
    )
    require(id_composite_index.index.is_unique, "ID composite metric keys are not unique.")
    ood_component_index = tables["ood_component"].set_index(
        ["model_key", "regime", "prediction_type", "component"]
    )
    require(ood_component_index.index.is_unique, "OOD component metric keys are not unique.")
    ood_composite_index = tables["ood_composite"].set_index(
        ["model_key", "regime", "prediction_type", "composite"]
    )
    require(ood_composite_index.index.is_unique, "OOD composite metric keys are not unique.")
    id_physical_index = tables["physical_fold"].set_index(
        ["model_key", "sample_size", "outer_fold", "prediction_type"]
    )
    require(id_physical_index.index.is_unique, "ID physical metric keys are not unique.")
    ood_physical_index = tables["ood_physical"].set_index(
        ["model_key", "regime", "prediction_type"]
    )
    require(ood_physical_index.index.is_unique, "OOD physical metric keys are not unique.")

    def verify_frame(
        frame: pd.DataFrame,
        *,
        sigma: np.ndarray,
        stored_metric_rows: dict[str, pd.Series],
        stored_component_rows: dict[tuple[str, str], pd.Series],
        stored_composite_rows: dict[tuple[str, str], pd.Series],
        stored_physical_rows: dict[str, pd.Series],
    ) -> None:
        nonlocal maximum_metric_difference, maximum_diagnostic_difference
        nonlocal maximum_component_difference, maximum_composite_difference
        truth = frame[[f"true_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        influent = frame[[f"influent_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        raw = frame[[f"raw_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        projected = frame[[f"projected_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        require(np.all(np.isfinite(truth)) and np.all(np.isfinite(raw)) and np.all(np.isfinite(projected)), "Prediction file contains non-finite values.")
        require(np.all(sigma > np.finfo(float).eps), "A training-target scale is non-positive.")
        for kind, values in (("raw", raw), ("projected", projected)):
            computed_metrics = normalized_metrics(truth, values, sigma)
            stored_metrics = stored_metric_rows[kind]
            for metric, value in computed_metrics.items():
                difference = abs(value - float(stored_metrics[metric]))
                maximum_metric_difference = max(maximum_metric_difference, difference)
                require(np.isclose(value, float(stored_metrics[metric]), rtol=1e-10, atol=1e-12), f"Stored {kind} {metric} does not reproduce.")

            residual = np.linalg.norm((invariant @ values.T - invariant @ influent.T).T, axis=1)
            negative_l1 = np.sum(np.abs(np.minimum(values / sigma, 0.0)), axis=1)
            minimum = np.min(values, axis=1)
            for column, calculated in (
                (f"{kind}_conservation_l2", residual),
                (f"{kind}_negative_standardized_l1", negative_l1),
                (f"{kind}_minimum_component", minimum),
            ):
                observed = frame[column].to_numpy(float)
                difference = float(np.max(np.abs(calculated - observed))) if len(frame) else 0.0
                maximum_diagnostic_difference = max(maximum_diagnostic_difference, difference)
                require(np.allclose(calculated, observed, rtol=1e-10, atol=1e-12), f"Stored diagnostic {column} does not reproduce.")
            require(np.array_equal(frame[f"{kind}_mass_violation"].to_numpy(bool), residual > tau), f"Stored {kind} mass flags do not reproduce.")
            require(np.array_equal(frame[f"{kind}_nonnegative_violation"].to_numpy(bool), minimum < -tau), f"Stored {kind} non-negativity flags do not reproduce.")

            physical_values = {
                "conservation_l2_mean": float(np.mean(residual)),
                "conservation_l2_std": float(np.std(residual, ddof=1)),
                "conservation_l2_median": float(np.median(residual)),
                "conservation_l2_q95": float(np.quantile(residual, 0.95)),
                "conservation_l2_max": float(np.max(residual)),
                "negative_standardized_l1_mean": float(np.mean(negative_l1)),
                "negative_standardized_l1_q95": float(np.quantile(negative_l1, 0.95)),
                "mass_violation_rate": float(np.mean(residual > tau)),
                "nonnegative_violation_rate": float(np.mean(minimum < -tau)),
            }
            stored_physical = stored_physical_rows[kind]
            for metric, value in physical_values.items():
                if metric not in stored_physical.index:
                    continue
                difference = abs(value - float(stored_physical[metric]))
                maximum_diagnostic_difference = max(maximum_diagnostic_difference, difference)
                require(np.isclose(value, float(stored_physical[metric]), rtol=1e-10, atol=1e-12), f"Stored physical summary {kind}/{metric} does not reproduce.")

            error = values - truth
            standardized_error = error / sigma
            displacement_components = projected - raw
            component_r2_denominator = np.sum(np.square(truth - np.mean(truth, axis=0)), axis=0)
            require(np.all(component_r2_denominator > 0), "A component target is constant.")
            component_r2 = 1.0 - np.sum(np.square(error), axis=0) / component_r2_denominator
            for index, component in enumerate(STATE_COLUMNS):
                negative = np.minimum(values[:, index], 0.0)
                calculated = {
                    "MSE": float(np.mean(np.square(error[:, index]))),
                    "RMSE": float(np.sqrt(np.mean(np.square(error[:, index])))),
                    "MAE": float(np.mean(np.abs(error[:, index]))),
                    "bias": float(np.mean(error[:, index])),
                    "maximum_absolute_error": float(np.max(np.abs(error[:, index]))),
                    "R2": float(component_r2[index]),
                    "nMSE_component": float(np.mean(np.square(standardized_error[:, index]))),
                    "nRMSE_component": float(np.sqrt(np.mean(np.square(standardized_error[:, index])))),
                    "nMAE_component": float(np.mean(np.abs(standardized_error[:, index]))),
                    "negative_count": int(np.sum(values[:, index] < 0.0)),
                    "negative_magnitude_mean": float(np.mean(np.abs(negative))),
                    "projection_displacement_mean": float(np.mean(np.abs(displacement_components[:, index]))) if kind == "projected" else np.nan,
                    "projection_displacement_standardized_mean": float(np.mean(np.abs(displacement_components[:, index]) / sigma[index])) if kind == "projected" else np.nan,
                }
                stored_component = stored_component_rows[(kind, component)]
                for metric, value in calculated.items():
                    stored_value = float(stored_component[metric])
                    if np.isnan(value):
                        require(np.isnan(stored_value), f"Stored raw component displacement is not NaN for {component}.")
                        continue
                    difference = abs(value - stored_value)
                    maximum_component_difference = max(maximum_component_difference, difference)
                    require(np.isclose(value, stored_value, rtol=1e-10, atol=1e-12), f"Stored component metric {kind}/{component}/{metric} does not reproduce.")

            true_composite = truth @ composition.T
            predicted_composite = values @ composition.T
            composite_error = predicted_composite - true_composite
            composite_denominator = np.sum(np.square(true_composite - np.mean(true_composite, axis=0)), axis=0)
            require(np.all(composite_denominator > 0), "A composite target is constant.")
            composite_r2 = 1.0 - np.sum(np.square(composite_error), axis=0) / composite_denominator
            for index, composite_name in enumerate(("COD", "TN", "TP", "TSS")):
                calculated = {
                    "MSE": float(np.mean(np.square(composite_error[:, index]))),
                    "RMSE": float(np.sqrt(np.mean(np.square(composite_error[:, index])))),
                    "MAE": float(np.mean(np.abs(composite_error[:, index]))),
                    "bias": float(np.mean(composite_error[:, index])),
                    "maximum_absolute_error": float(np.max(np.abs(composite_error[:, index]))),
                    "R2": float(composite_r2[index]),
                }
                stored_composite = stored_composite_rows[(kind, composite_name)]
                for metric, value in calculated.items():
                    difference = abs(value - float(stored_composite[metric]))
                    maximum_composite_difference = max(maximum_composite_difference, difference)
                    require(np.isclose(value, float(stored_composite[metric]), rtol=1e-10, atol=1e-12), f"Stored composite metric {kind}/{composite_name}/{metric} does not reproduce.")
        displacement = np.linalg.norm((projected - raw) / sigma, axis=1)
        observed_displacement = frame["standardized_displacement_l2"].to_numpy(float)
        displacement_difference = float(np.max(np.abs(displacement - observed_displacement))) if len(frame) else 0.0
        maximum_diagnostic_difference = max(maximum_diagnostic_difference, displacement_difference)
        require(np.allclose(displacement, observed_displacement, rtol=1e-10, atol=1e-12), "Stored projection displacement does not reproduce.")
        alpha = frame["backtracking_alpha"].to_numpy(float)
        require(np.isfinite(alpha).all() and np.all((alpha >= 0.0) & (alpha <= 1.0)), "Backtracking alpha is outside [0, 1].")
        require(np.array_equal(frame["backtracking_active"].to_numpy(bool), alpha < 1.0), "Backtracking flags do not match alpha.")
        shared_physical = {
            "standardized_displacement_l2_mean": float(np.mean(displacement)),
            "backtracking_rate": float(np.mean(alpha < 1.0)),
            "backtracking_alpha_mean": float(np.mean(alpha)),
            "backtracking_alpha_min": float(np.min(alpha)),
        }
        for kind in ("raw", "projected"):
            stored_physical = stored_physical_rows[kind]
            for metric, value in shared_physical.items():
                difference = abs(value - float(stored_physical[metric]))
                maximum_diagnostic_difference = max(maximum_diagnostic_difference, difference)
                require(np.isclose(value, float(stored_physical[metric]), rtol=1e-10, atol=1e-12), f"Stored shared physical summary {kind}/{metric} does not reproduce.")

    required_prediction_columns = [
        "sample_id",
        *[f"true_{name}" for name in STATE_COLUMNS],
        *[f"raw_{name}" for name in STATE_COLUMNS],
        *[f"projected_{name}" for name in STATE_COLUMNS],
        *[f"influent_{name}" for name in STATE_COLUMNS],
        "raw_conservation_l2", "projected_conservation_l2",
        "raw_negative_standardized_l1", "projected_negative_standardized_l1",
        "raw_mass_violation", "projected_mass_violation",
        "raw_nonnegative_violation", "projected_nonnegative_violation",
        "standardized_displacement_l2", "backtracking_alpha", "backtracking_active",
        "raw_minimum_component", "projected_minimum_component",
    ]

    for path in id_paths:
        match = re.fullmatch(r"(.+)_n(\d+)_fold(\d+)", path.stem)
        require(match is not None, f"Unexpected ID prediction filename: {path.name}")
        model_key, size_text, fold_text = match.groups()
        size, fold = int(size_text), int(fold_text)
        frame = pd.read_parquet(path, columns=required_prediction_columns)
        frame["sample_id"] = frame["sample_id"].astype(str)
        require(frame["sample_id"].is_unique, f"Duplicate sample IDs in {path.name}.")
        subset = splits[splits["master_position"] < size]
        expected_ids = set(subset.loc[subset["outer_fold"] == fold, "sample_id"])
        require(set(frame["sample_id"]) == expected_ids, f"Scored IDs disagree with the master split in {path.name}.")
        expected_truth = id_by_sample.loc[frame["sample_id"], [f"Out_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        observed_truth = frame[[f"true_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        require(np.array_equal(expected_truth, observed_truth), f"Stored ground truth differs from the ID dataset in {path.name}.")
        expected_influent = id_by_sample.loc[frame["sample_id"], [f"In_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        observed_influent = frame[[f"influent_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        require(np.array_equal(expected_influent, observed_influent), f"Stored influent differs from the ID dataset in {path.name}.")
        train_ids = subset.loc[subset["outer_fold"] != fold, "sample_id"]
        sigma = id_by_sample.loc[train_ids, [f"Out_{name}" for name in STATE_COLUMNS]].to_numpy(float).std(axis=0, ddof=0)
        stored_rows = {
            kind: id_metric_index.loc[(model_key, size, fold, kind)]
            for kind in ("raw", "projected")
        }
        stored_component_rows = {
            (kind, component): id_component_index.loc[(model_key, size, fold, kind, component)]
            for kind in ("raw", "projected") for component in STATE_COLUMNS
        }
        stored_composite_rows = {
            (kind, composite_name): id_composite_index.loc[(model_key, size, fold, kind, composite_name)]
            for kind in ("raw", "projected") for composite_name in ("COD", "TN", "TP", "TSS")
        }
        stored_physical_rows = {
            kind: id_physical_index.loc[(model_key, size, fold, kind)]
            for kind in ("raw", "projected")
        }
        verify_frame(
            frame,
            sigma=sigma,
            stored_metric_rows=stored_rows,
            stored_component_rows=stored_component_rows,
            stored_composite_rows=stored_composite_rows,
            stored_physical_rows=stored_physical_rows,
        )
        count = len(frame)
        total_instances += count
        id_instances += count
        max_conservation = max(max_conservation, float(frame["projected_conservation_l2"].max()))
        min_component = min(min_component, float(frame["projected_minimum_component"].min()))
        mass_violations += int(frame["projected_mass_violation"].sum())
        nonnegative_violations += int(frame["projected_nonnegative_violation"].sum())

    full_sigma = id_data[[f"Out_{name}" for name in STATE_COLUMNS]].to_numpy(float).std(axis=0, ddof=0)
    ood_by_regime: dict[str, pd.DataFrame] = {}
    for regime in ("mild", "severe"):
        ood_data = pd.read_parquet(
            run_root / "datasets" / f"ood_{regime}.parquet",
            columns=[
                "sample_id",
                *[f"In_{name}" for name in STATE_COLUMNS],
                *[f"Out_{name}" for name in STATE_COLUMNS],
            ],
        )
        ood_data["sample_id"] = ood_data["sample_id"].astype(str)
        indexed = ood_data.set_index("sample_id")
        require(indexed.index.is_unique, f"{regime} OOD sample IDs are not unique.")
        ood_by_regime[regime] = indexed
    for path in ood_paths:
        regime = next((name for name in ("mild", "severe") if path.stem.endswith(f"_{name}")), None)
        require(regime is not None, f"Unexpected OOD prediction filename: {path.name}")
        model_key = path.stem[: -(len(regime) + 1)]
        require(model_key in MODEL_ORDER, f"Unknown model in OOD prediction file: {path.name}")
        frame = pd.read_parquet(path, columns=required_prediction_columns)
        frame["sample_id"] = frame["sample_id"].astype(str)
        require(frame["sample_id"].is_unique, f"Duplicate sample IDs in {path.name}.")
        ood_by_sample = ood_by_regime[regime]
        require(set(frame["sample_id"]) == set(ood_by_sample.index), f"Scored IDs disagree with the {regime} OOD dataset in {path.name}.")
        expected_truth = ood_by_sample.loc[frame["sample_id"], [f"Out_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        observed_truth = frame[[f"true_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        require(np.array_equal(expected_truth, observed_truth), f"Stored ground truth differs from the OOD dataset in {path.name}.")
        expected_influent = ood_by_sample.loc[frame["sample_id"], [f"In_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        observed_influent = frame[[f"influent_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        require(np.array_equal(expected_influent, observed_influent), f"Stored influent differs from the OOD dataset in {path.name}.")
        stored_rows = {
            kind: ood_metric_index.loc[(model_key, regime, kind)]
            for kind in ("raw", "projected")
        }
        stored_component_rows = {
            (kind, component): ood_component_index.loc[(model_key, regime, kind, component)]
            for kind in ("raw", "projected") for component in STATE_COLUMNS
        }
        stored_composite_rows = {
            (kind, composite_name): ood_composite_index.loc[(model_key, regime, kind, composite_name)]
            for kind in ("raw", "projected") for composite_name in ("COD", "TN", "TP", "TSS")
        }
        stored_physical_rows = {
            kind: ood_physical_index.loc[(model_key, regime, kind)]
            for kind in ("raw", "projected")
        }
        verify_frame(
            frame,
            sigma=full_sigma,
            stored_metric_rows=stored_rows,
            stored_component_rows=stored_component_rows,
            stored_composite_rows=stored_composite_rows,
            stored_physical_rows=stored_physical_rows,
        )
        count = len(frame)
        total_instances += count
        max_conservation = max(max_conservation, float(frame["projected_conservation_l2"].max()))
        min_component = min(min_component, float(frame["projected_minimum_component"].min()))
        mass_violations += int(frame["projected_mass_violation"].sum())
        nonnegative_violations += int(frame["projected_nonnegative_violation"].sum())
    if strict_full:
        require(id_instances == 750_750, f"Expected 750,750 ID prediction instances, found {id_instances}.")
        require(total_instances == 766_350, f"Expected 766,350 total prediction instances, found {total_instances}.")
    require(max_conservation <= tau, f"Projected conservation residual {max_conservation:.6g} exceeds tau={tau:.6g}.")
    require(min_component >= -tau, f"Projected minimum {min_component:.6g} is below -tau={-tau:.6g}.")
    require(mass_violations == 0, f"Found {mass_violations} projected mass-violation instances.")
    require(nonnegative_violations == 0, f"Found {nonnegative_violations} projected non-negativity violations.")
    return {
        "id_prediction_instances": id_instances,
        "ood_prediction_instances": total_instances - id_instances,
        "total_prediction_instances": total_instances,
        "projected_mass_violation_instances": mass_violations,
        "projected_nonnegative_violation_instances": nonnegative_violations,
        "maximum_projected_conservation_l2": max_conservation,
        "minimum_projected_component": min_component,
        "maximum_recomputed_metric_absolute_difference": maximum_metric_difference,
        "maximum_recomputed_diagnostic_absolute_difference": maximum_diagnostic_difference,
        "maximum_recomputed_component_metric_absolute_difference": maximum_component_difference,
        "maximum_recomputed_composite_metric_absolute_difference": maximum_composite_difference,
        "tolerance": tau,
    }


def attempt_summary(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, stem in (
        ("In-distribution", "id"),
        ("Mild out-of-distribution", "ood_mild"),
        ("Severe out-of-distribution", "ood_severe"),
    ):
        attempts = pd.read_parquet(run_root / "datasets" / f"{stem}.attempts.parquet")
        retained = int(attempts["selected_for_dataset"].fillna(False).astype(bool).sum())
        solver_accepted = int(attempts["accepted"].fillna(False).astype(bool).sum())
        solver_failed = int((~attempts["accepted"].fillna(False).astype(bool)).sum())
        rows.append(
            {
                "set": label,
                "attempted": len(attempts),
                "solver_accepted": solver_accepted,
                "solver_failed": solver_failed,
                "retained": retained,
                "valid_unselected": solver_accepted - retained,
                "retention_rate": retained / len(attempts),
                "solver_acceptance_rate": solver_accepted / len(attempts),
            }
        )
    return pd.DataFrame(rows)


def validate_datasets_and_splits(
    run_root: Path,
    resolved: dict[str, Any],
    *,
    strict_full: bool,
) -> dict[str, Any]:
    operators = np.load(run_root / "matrices" / "operators.npz")
    invariant = np.asarray(operators["invariant"], dtype=float)
    petersen = np.asarray(operators["petersen"], dtype=float)
    invariant_error = float(np.max(np.abs(invariant @ petersen.T)))
    matrix_validation = load_json(run_root / "matrices" / "validation.json")
    require(np.isclose(invariant_error, float(matrix_validation["max_abs_A_nu_transpose"]), rtol=1e-12, atol=1e-18), "Invariant-matrix validation does not reproduce.")

    expected_counts = {
        "id": int(resolved["simulation"]["sampling"]["id_samples"]),
        "ood_mild": int(resolved["simulation"]["ood"]["regimes"]["mild"]["n_samples"]),
        "ood_severe": int(resolved["simulation"]["ood"]["regimes"]["severe"]["n_samples"]),
    }
    if strict_full:
        require(expected_counts == {"id": 10_000, "ood_mild": 600, "ood_severe": 600}, "Resolved dataset counts are not the full contract.")
    validation_rows: list[dict[str, Any]] = []
    ood_distance_by_regime: dict[str, np.ndarray] = {}
    ood_validation = pd.read_csv(run_root / "datasets" / "ood_validation.csv").set_index("set")
    for stem, expected in expected_counts.items():
        data = pd.read_parquet(run_root / "datasets" / f"{stem}.parquet")
        attempts = pd.read_parquet(run_root / "datasets" / f"{stem}.attempts.parquet")
        require(len(data) == expected, f"{stem} dataset has {len(data)} rather than {expected} rows.")
        require(data["sample_id"].astype(str).is_unique, f"{stem} sample IDs are not unique.")
        require(data["candidate_id"].is_unique, f"{stem} retained candidate IDs are not unique.")
        selected = attempts[attempts["selected_for_dataset"].fillna(False).astype(bool)].copy()
        require(len(selected) == expected, f"{stem} has {len(selected)} selected attempts rather than {expected}.")
        require(selected["accepted"].fillna(False).astype(bool).all(), f"{stem} includes a selected solver-failed attempt.")
        require(set(selected["accepted_sample_id"].astype(str)) == set(data["sample_id"].astype(str)), f"{stem} selected attempt IDs do not match the dataset.")
        require(set(selected["candidate_id"].astype(int)) == set(data["candidate_id"].astype(int)), f"{stem} selected candidate IDs do not match the dataset.")
        influent = data[[f"In_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        output = data[[f"Out_{name}" for name in STATE_COLUMNS]].to_numpy(float)
        residual = np.linalg.norm((invariant @ output.T - invariant @ influent.T).T, axis=1)
        maximum_residual = float(residual.max())
        minimum_component = float(output.min())
        require(maximum_residual <= float(resolved["projection"]["tau"]), f"{stem} ground truth violates conservation tolerance.")
        require(minimum_component >= -float(resolved["projection"]["tau"]), f"{stem} ground truth violates non-negativity tolerance.")
        if stem == "id":
            stored = load_json(run_root / "datasets" / "id.validation.json")
            stored_residual = float(stored["maximum_ground_truth_conservation_l2"])
            stored_minimum = float(stored["minimum_ground_truth_component"])
            stored_accepted = int(stored["accepted"])
            stored_attempted = int(stored["attempted"])
        else:
            regime = stem.removeprefix("ood_")
            require(regime in ood_validation.index, f"Missing OOD validation row for {regime}.")
            stored = ood_validation.loc[regime]
            stored_residual = float(stored["maximum_conservation_l2"])
            stored_minimum = float(stored["minimum_component"])
            stored_accepted = int(stored["accepted"])
            stored_attempted = int(stored["attempted"])
            allowed_perturbations = set(resolved["simulation"]["ood"]["regimes"][regime]["ranges"])
            variable_ranges = {
                **resolved["simulation"]["operational_ranges"],
                **resolved["simulation"]["influent_state_ranges"],
            }
            variable_columns = {
                name: name if name in {"HRT", "Aeration"} else f"In_{name}"
                for name in variable_ranges
            }
            for design_frame, design_name in ((data, "retained"), (attempts, "attempted")):
                parsed = design_frame["perturbed_variables"].astype(str).map(
                    lambda value: tuple(part for part in value.split(";") if part)
                )
                require(parsed.map(len).isin([1, 2]).all(), f"{regime} {design_name} rows do not perturb one or two variables.")
                require(parsed.map(lambda values: len(values) == len(set(values)) and set(values) <= allowed_perturbations).all(), f"{regime} {design_name} rows contain an invalid perturbed variable.")
                for row_position, selected_names in enumerate(parsed):
                    selected_set = set(selected_names)
                    for variable, (lower, upper) in variable_ranges.items():
                        observed = float(design_frame.iloc[row_position][variable_columns[variable]])
                        if variable in selected_set:
                            ood_lower, ood_upper = resolved["simulation"]["ood"]["regimes"][regime]["ranges"][variable]
                            require(float(ood_lower) <= observed <= float(ood_upper), f"{regime} {design_name} perturbed value is outside its OOD interval.")
                        else:
                            require(float(lower) <= observed <= float(upper), f"{regime} {design_name} non-perturbed value left the ID interval.")
            input_names = ["HRT", "Aeration", *resolved["simulation"]["workbook"]["state_columns"]]
            input_columns = [variable_columns[name] for name in input_names]
            lower = np.asarray([variable_ranges[name][0] for name in input_names], dtype=float)
            upper = np.asarray([variable_ranges[name][1] for name in input_names], dtype=float)
            span = upper - lower
            values = data[input_columns].to_numpy(float)
            exterior = np.maximum.reduce([(lower - values) / span, (values - upper) / span, np.zeros_like(values)])
            normalized_distance = np.linalg.norm(exterior, axis=1)
            require(np.all(normalized_distance > 0), f"{regime} contains an OOD row with zero exterior distance.")
            ood_distance_by_regime[regime] = normalized_distance
            one_candidate_rate = float(attempts["perturbed_variables"].astype(str).str.count(";").eq(0).mean())
            two_candidate_rate = float(attempts["perturbed_variables"].astype(str).str.count(";").eq(1).mean())
            one_retained_rate = float(data["perturbed_variables"].astype(str).str.count(";").eq(0).mean())
            two_retained_rate = float(data["perturbed_variables"].astype(str).str.count(";").eq(1).mean())
            for field, value in (
                ("one_variable_candidate_rate", one_candidate_rate),
                ("two_variable_candidate_rate", two_candidate_rate),
                ("one_variable_accepted_rate", one_retained_rate),
                ("two_variable_accepted_rate", two_retained_rate),
            ):
                require(np.isclose(value, float(stored[field]), rtol=1e-12, atol=1e-15), f"{regime} {field} does not reproduce.")
        require(stored_accepted == expected and stored_attempted == len(attempts), f"{stem} validation counts disagree.")
        require(np.isclose(maximum_residual, stored_residual, rtol=1e-12, atol=1e-15), f"{stem} maximum invariant residual does not reproduce.")
        require(np.isclose(minimum_component, stored_minimum, rtol=1e-12, atol=1e-15), f"{stem} minimum component does not reproduce.")
        validation_rows.append({"set": stem, "attempted": len(attempts), "retained": expected, "maximum_conservation_l2": maximum_residual, "minimum_component": minimum_component})

    splits = pd.read_parquet(run_root / "splits" / "master_assignments.parquet")
    id_data = pd.read_parquet(run_root / "datasets" / "id.parquet", columns=["sample_id"])
    require(len(splits) == len(id_data), "Master split row count differs from the ID dataset.")
    require(splits["sample_id"].astype(str).is_unique, "Master split sample IDs are not unique.")
    require(splits["master_position"].is_unique, "Master split positions are not unique.")
    require(set(splits["master_position"].astype(int)) == set(range(len(id_data))), "Master split positions are not contiguous.")
    require(set(splits["sample_id"].astype(str)) == set(id_data["sample_id"].astype(str)), "Master split IDs differ from the ID dataset.")
    outer_folds = int(resolved["evaluation"]["nested_cv"]["outer_folds"])
    require(set(splits["outer_fold"].astype(int)) == set(range(outer_folds)), "Master split outer-fold IDs are incomplete.")
    for size in resolved["evaluation"]["nested_cv"]["sample_sizes"]:
        subset = splits[splits["master_position"] < int(size)]
        counts = subset["outer_fold"].value_counts().reindex(range(outer_folds), fill_value=0).to_numpy(int)
        require(np.array_equal(counts, np.full(outer_folds, int(size) // outer_folds)), f"Nested size {size} is not exactly outer-fold balanced.")
    require(float(np.min(ood_distance_by_regime["severe"])) > float(np.max(ood_distance_by_regime["mild"])), "Severe OOD distances do not lie beyond the mild stratum.")
    return {
        "maximum_abs_A_nu_transpose": invariant_error,
        "datasets": validation_rows,
        "ood_normalized_exterior_distance": {
            regime: {
                "minimum": float(values.min()),
                "mean": float(values.mean()),
                "maximum": float(values.max()),
            }
            for regime, values in ood_distance_by_regime.items()
        },
        "master_split_rows": len(splits),
    }


def validate_timing_repetitions(
    run_root: Path,
    resolved: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    *,
    strict_full: bool,
) -> dict[str, Any]:
    batch_size = int(resolved["evaluation"]["timing"]["batch_size"])
    measured_runs = int(resolved["evaluation"]["timing"]["measured_runs"])
    id_index = tables["timing_fold"].set_index(["model_key", "sample_size", "outer_fold"])
    ood_index = tables["ood_timing"].set_index(["model_key", "regime"])
    require(id_index.index.is_unique and ood_index.index.is_unique, "Timing summary keys are not unique.")
    expected_id_paths = {
        run_root / "timing" / f"{key}_n{size}_fold{fold}.parquet"
        for key in MODEL_ORDER
        for size in sorted(tables["timing_fold"]["sample_size"].unique())
        for fold in sorted(tables["timing_fold"]["outer_fold"].unique())
    }
    expected_ood_paths = {
        run_root / "timing" / f"ood_{key}_{regime}.parquet"
        for key in MODEL_ORDER for regime in ("mild", "severe")
    }
    actual_paths = set((run_root / "timing").glob("*.parquet"))
    require(actual_paths == expected_id_paths | expected_ood_paths, "Timing-repetition file grid is incomplete or contains extras.")
    if strict_full:
        require(len(expected_id_paths) == 715 and len(expected_ood_paths) == 26, "Full timing file cardinality is not 715 + 26.")
    maximum_difference = 0.0

    def check(path: Path, stored: pd.Series, metadata: dict[str, Any]) -> None:
        nonlocal maximum_difference
        frame = pd.read_parquet(path)
        require(len(frame) == measured_runs, f"{path.name} has {len(frame)} rather than {measured_runs} repetitions.")
        require(set(frame["repetition"].astype(int)) == set(range(measured_runs)), f"{path.name} repetition IDs are incomplete.")
        seconds = frame[["raw_seconds", "projection_seconds", "end_to_end_seconds"]].to_numpy(float)
        require(np.isfinite(seconds).all() and np.all(seconds >= 0.0), f"{path.name} contains an invalid duration.")
        for column, expected in metadata.items():
            require(frame[column].nunique(dropna=False) == 1 and str(frame[column].iloc[0]) == str(expected), f"{path.name} has inconsistent {column} metadata.")
        calculated = {
            "raw_latency_ms_per_sample": float(np.median(frame["raw_seconds"]) * 1000.0 / batch_size),
            "projection_latency_ms_per_sample": float(np.median(frame["projection_seconds"]) * 1000.0 / batch_size),
            "end_to_end_latency_ms_per_sample": float(np.median(frame["end_to_end_seconds"]) * 1000.0 / batch_size),
        }
        for field, value in calculated.items():
            difference = abs(value - float(stored[field]))
            maximum_difference = max(maximum_difference, difference)
            require(np.isclose(value, float(stored[field]), rtol=1e-10, atol=1e-12), f"{path.name} does not reproduce {field}.")

    for key, size, fold in id_index.index:
        check(
            run_root / "timing" / f"{key}_n{int(size)}_fold{int(fold)}.parquet",
            id_index.loc[(key, size, fold)],
            {
                "model_key": key,
                "sample_size": int(size),
                "outer_fold": int(fold),
                "split": "id_validation",
            },
        )
    for key, regime in ood_index.index:
        check(
            run_root / "timing" / f"ood_{key}_{regime}.parquet",
            ood_index.loc[(key, regime)],
            {"model_key": key, "regime": regime, "split": "ood"},
        )
    return {
        "id_timing_repetition_files": len(expected_id_paths),
        "ood_timing_repetition_files": len(expected_ood_paths),
        "measured_runs_per_file": measured_runs,
        "batch_size": batch_size,
        "maximum_recomputed_latency_absolute_difference": maximum_difference,
    }


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        metadata = (
            {"Creator": "ICSOR final paper asset builder", "CreationDate": None, "ModDate": None}
            if extension == "pdf"
            else {"Software": "ICSOR final paper asset builder"}
        )
        fig.savefig(
            output_dir / f"{stem}.{extension}",
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            metadata=metadata,
        )
    plt.close(fig)


def style() -> None:
    mpl.rcParams.update(
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
        }
    )


def plot_model_curve(axis: mpl.axes.Axes, rows: pd.DataFrame, y: str, y_sd: str) -> None:
    for key in MODEL_ORDER:
        subset = rows[rows["model_key"].astype(str) == key].sort_values("sample_size")
        x = subset["sample_size"].to_numpy(float)
        mean = subset[y].to_numpy(float)
        sd = subset[y_sd].fillna(0.0).to_numpy(float)
        axis.plot(x, mean, color=MODEL_COLORS[key], marker=MODEL_MARKERS[key], linewidth=1.0, markersize=3.5, label=MODEL_LABELS[key])
        lower = np.maximum(mean - sd, np.finfo(float).tiny) if axis.get_yscale() == "log" else mean - sd
        axis.fill_between(x, lower, mean + sd, color=MODEL_COLORS[key], alpha=0.07, linewidth=0)


def build_main_figures(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    raw = ordered(tables["id_summary"].query("prediction_type == 'raw'"))
    timing = ordered(tables["timing_summary"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.6), constrained_layout=True)
    plot_model_curve(axes[0], raw, "nRMSE_mean", "nRMSE_std")
    axes[0].set(xlabel="Nested sample total", ylabel="nRMSE", title="(a) Predictive error")
    axes[0].grid(axis="y", linestyle=":", alpha=0.3)
    axes[1].set_yscale("log")
    plot_model_curve(axes[1], timing, "setup_seconds_mean", "setup_seconds_std")
    axes[1].set(xlabel="Nested sample total", ylabel="Model setup time (s, log scale)", title="(b) Computational scaling")
    axes[1].grid(axis="y", which="both", linestyle=":", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=7.8)
    save_figure(fig, output_dir, "figure_main_scaling")

    pivot = tables["id_summary"].pivot(index=["model_key", "sample_size"], columns="prediction_type", values="nRMSE_mean").reset_index()
    pivot["delta_nRMSE"] = pivot["projected"] - pivot["raw"]
    sizes = sorted(pivot["sample_size"].unique())
    matrix = np.vstack([
        pivot[pivot["model_key"] == key].set_index("sample_size").reindex(sizes)["delta_nRMSE"].to_numpy(float)
        for key in MODEL_ORDER
    ])
    limit = max(float(np.nanmax(np.abs(matrix))), np.finfo(float).eps)
    fig, axis = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(sizes)), [f"{value:,}" for value in sizes], rotation=45, ha="right")
    axis.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[key] for key in MODEL_ORDER])
    axis.set(xlabel="Nested sample total", ylabel="Model")
    fig.colorbar(image, ax=axis, label=r"$\Delta$nRMSE (projected $-$ raw)", shrink=0.82)
    save_figure(fig, output_dir, "figure_main_projection_delta")

    severe = tables["ood"].query("regime == 'severe'").pivot(index="model_key", columns="prediction_type", values="nRMSE").reindex(MODEL_ORDER)
    y = np.arange(len(MODEL_ORDER))
    fig, axis = plt.subplots(figsize=(6.8, 5.3), constrained_layout=True)
    for position, key in enumerate(MODEL_ORDER):
        axis.plot([severe.loc[key, "raw"], severe.loc[key, "projected"]], [position, position], color="#9A9A9A", linewidth=0.8, zorder=1)
    axis.scatter(severe["raw"], y, marker="s", facecolors="none", edgecolors="#E76F51", label="Raw", zorder=2)
    axis.scatter(severe["projected"], y, marker="o", color="#2A9D8F", label="Projected", zorder=3)
    axis.set_yticks(y, [MODEL_LABELS[key] for key in MODEL_ORDER])
    axis.invert_yaxis()
    axis.set(xlabel="nRMSE", ylabel="Model")
    axis.grid(axis="x", linestyle=":", alpha=0.3)
    axis.legend(frameon=False)
    save_figure(fig, output_dir, "figure_main_severe_ood")


def heatmap_panel(
    axis: mpl.axes.Axes,
    matrix: np.ndarray,
    title: str,
    cmap: str,
    *,
    log10: bool = False,
    log_floor: float = 1e-16,
    norm: mpl.colors.Normalize | None = None,
) -> mpl.image.AxesImage:
    values = np.asarray(matrix, dtype=float)
    if log10:
        values = np.log10(np.maximum(values, log_floor))
    image = axis.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    axis.set_title(title)
    return image


def build_supplement_figures(run_root: Path, tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    source_dir = output_dir.parent / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    raw = ordered(tables["id_summary"].query("prediction_type == 'raw'"))
    raw.to_csv(source_dir / "figure_s_learning_curves.csv", index=False)
    fig, axis = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    plot_model_curve(axis, raw, "nRMSE_mean", "nRMSE_std")
    axis.set(xlabel="Nested sample total", ylabel="nRMSE")
    axis.grid(axis="y", linestyle=":", alpha=0.3)
    axis.legend(ncol=3, fontsize=7.8, frameon=False)
    save_figure(fig, output_dir, "figure_s_learning_curves")

    full_size = int(tables["component"]["sample_size"].max())
    component = tables["component"].query("sample_size == @full_size")
    component_mean = component.groupby(["model_key", "component", "prediction_type"], sort=False).agg(
        nRMSE=("nRMSE_component", "mean"),
        RMSE=("RMSE", "mean"),
        displacement_standardized=("projection_displacement_standardized_mean", "mean"),
        displacement_physical=("projection_displacement_mean", "mean"),
    ).reset_index()
    ordered(component_mean).to_csv(source_dir / "figure_s_component_atlas.csv", index=False)
    component_order = [name for name in STATE_COLUMNS if name in set(component_mean["component"])]
    raw_matrix = component_mean.query("prediction_type == 'raw'").pivot(index="model_key", columns="component", values="nRMSE").reindex(index=MODEL_ORDER, columns=component_order).to_numpy(float)
    projected_matrix = component_mean.query("prediction_type == 'projected'").pivot(index="model_key", columns="component", values="nRMSE").reindex(index=MODEL_ORDER, columns=component_order).to_numpy(float)
    displacement_standardized = component_mean.query("prediction_type == 'projected'").pivot(index="model_key", columns="component", values="displacement_standardized").reindex(index=MODEL_ORDER, columns=component_order).to_numpy(float)
    standardized_norm = mpl.colors.Normalize(vmin=float(np.nanmin([raw_matrix, projected_matrix])), vmax=float(np.nanmax([raw_matrix, projected_matrix])))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 5.6), constrained_layout=True)
    images = [
        heatmap_panel(axes[0], raw_matrix, "(a) Raw component nRMSE", "viridis", norm=standardized_norm),
        heatmap_panel(axes[1], projected_matrix, "(b) Projected component nRMSE", "viridis", norm=standardized_norm),
        heatmap_panel(axes[2], displacement_standardized, "(c) Standardized displacement", "magma"),
    ]
    for axis in axes:
        axis.set_xticks(range(len(component_order)), component_order, rotation=60, ha="right")
        axis.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[key] for key in MODEL_ORDER])
    fig.colorbar(images[0], ax=[axes[0], axes[1]], shrink=0.72)
    fig.colorbar(images[2], ax=axes[2], shrink=0.72)
    save_figure(fig, output_dir, "figure_s_component_atlas")

    physical = tables["physical_summary"]
    ordered(physical).to_csv(source_dir / "figure_s_physical_diagnostics.csv", index=False)
    sizes = sorted(physical["sample_size"].unique())
    specs = [
        ("raw", "conservation_l2_mean_mean", "(a) log10 raw conservation residual (floor 1e-16)", "viridis", True, 1.0),
        ("raw", "nonnegative_violation_rate_mean", "(b) Raw non-negativity violation (%)", "magma", False, 100.0),
        ("raw", "negative_standardized_l1_mean_mean", "(c) Raw standardized negative magnitude", "inferno", False, 1.0),
        ("projected", "backtracking_rate_mean", "(d) Backtracking activation (%)", "cividis", False, 100.0),
        ("projected", "standardized_displacement_l2_mean_mean", "(e) Standardized displacement", "plasma", False, 1.0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 7.4), constrained_layout=True)
    for axis, (prediction, value, title, cmap, log10, factor) in zip(axes.flat, specs):
        subset = physical.query("prediction_type == @prediction").copy()
        subset[value] = subset[value] * factor
        matrix = subset.pivot(index="model_key", columns="sample_size", values=value).reindex(index=MODEL_ORDER, columns=sizes).to_numpy(float)
        image = heatmap_panel(axis, matrix, title, cmap, log10=log10)
        axis.set_xticks(range(len(sizes)), [f"{size:,}" for size in sizes], rotation=45, ha="right")
        axis.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[key] for key in MODEL_ORDER])
        fig.colorbar(image, ax=axis, shrink=0.75)
    axes.flat[-1].axis("off")
    save_figure(fig, output_dir, "figure_s_physical_diagnostics")

    id_data = pd.read_parquet(run_root / "datasets" / "id.parquet", columns=[f"Out_{name}" for name in STATE_COLUMNS])
    target_sd = id_data.std(axis=0, ddof=0).to_numpy(float)
    require(np.all(target_sd > 0), "A full-ID target standard deviation is non-positive.")
    distribution_rows: list[pd.DataFrame] = []
    for key in MODEL_ORDER:
        for regime in ("mild", "severe"):
            prediction = pd.read_parquet(run_root / "predictions" / "ood" / f"{key}_{regime}.parquet")
            truth = prediction[[f"true_{name}" for name in STATE_COLUMNS]].to_numpy(float)
            for kind in ("raw", "projected"):
                estimate = prediction[[f"{kind}_{name}" for name in STATE_COLUMNS]].to_numpy(float)
                per_sample = np.sqrt(np.mean(np.square((estimate - truth) / target_sd), axis=1))
                distribution_rows.append(
                    pd.DataFrame(
                        {
                            "sample_id": prediction["sample_id"].astype(str).to_numpy(),
                            "model_key": key,
                            "regime": regime,
                            "prediction_type": kind,
                            "sample_nRMSE": per_sample,
                            "displacement": prediction["standardized_displacement_l2"].to_numpy(float),
                        }
                    )
                )
    distributions = pd.concat(distribution_rows, ignore_index=True)
    ordered(distributions).to_csv(source_dir / "figure_s_ood_distributions.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=True)
    positions = np.arange(len(MODEL_ORDER))
    for column, regime in enumerate(("mild", "severe")):
        axis = axes[0, column]
        for offset, kind, color in ((-0.18, "raw", "#E76F51"), (0.18, "projected", "#2A9D8F")):
            samples = [distributions.query("regime == @regime and prediction_type == @kind and model_key == @key")["sample_nRMSE"].to_numpy(float) for key in MODEL_ORDER]
            box = axis.boxplot(samples, positions=positions + offset, widths=0.30, whis=(5, 95), showfliers=False, patch_artist=True)
            for patch in box["boxes"]:
                patch.set(facecolor="white", edgecolor=color, linewidth=0.8)
            for item in box["medians"]:
                item.set(color=color, linewidth=1.0)
            for item in [*box["whiskers"], *box["caps"]]:
                item.set(color=color, linewidth=0.7)
        axis.set_title(f"({chr(97 + column)}) {regime.capitalize()} per-sample nRMSE")
        axis.set_ylabel("Per-sample nRMSE")
        axis.grid(axis="y", linestyle=":", alpha=0.3)
        axis.legend(handles=[Line2D([0], [0], color="#E76F51", label="Raw"), Line2D([0], [0], color="#2A9D8F", label="Projected")], frameon=False)
        displacement = [
            distributions.query(
                "regime == @regime and prediction_type == 'raw' and model_key == @key"
            )["displacement"].to_numpy(float)
            for key in MODEL_ORDER
        ]
        axes[1, column].boxplot(displacement, positions=positions, widths=0.55, whis=(5, 95), showfliers=False)
        axes[1, column].set_title(f"({chr(99 + column)}) {regime.capitalize()} projection displacement")
        axes[1, column].set_ylabel("Standardized displacement")
        axes[1, column].grid(axis="y", linestyle=":", alpha=0.3)
    for axis in axes.flat:
        axis.set_xticks(positions, [MODEL_LABELS[key] for key in MODEL_ORDER], rotation=45, ha="right")
    save_figure(fig, output_dir, "figure_s_ood_distributions")


def export_analysis_tables(run_root: Path, tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts = attempt_summary(run_root)
    attempts.to_csv(output_dir / "dataset_attempt_summary.csv", index=False)
    id_validation = load_json(run_root / "datasets" / "id.validation.json")
    ood_validation = pd.read_csv(run_root / "datasets" / "ood_validation.csv").set_index("set")
    validation_values = pd.DataFrame(
        [
            {
                "set": "In-distribution",
                "maximum_conservation_l2": id_validation["maximum_ground_truth_conservation_l2"],
                "minimum_component": id_validation["minimum_ground_truth_component"],
            },
            {
                "set": "Mild out-of-distribution",
                "maximum_conservation_l2": ood_validation.loc["mild", "maximum_conservation_l2"],
                "minimum_component": ood_validation.loc["mild", "minimum_component"],
            },
            {
                "set": "Severe out-of-distribution",
                "maximum_conservation_l2": ood_validation.loc["severe", "maximum_conservation_l2"],
                "minimum_component": ood_validation.loc["severe", "minimum_component"],
            },
        ]
    )
    dataset_validation = attempts.merge(validation_values, on="set", validate="one_to_one")
    dataset_validation.to_csv(output_dir / "main_dataset_validation.csv", index=False)

    for name, frame in tables.items():
        ordered(frame).to_csv(output_dir / f"complete_{name}.csv", index=False)

    largest = int(tables["id_summary"]["sample_size"].max())
    id_pivot = tables["id_summary"].query("sample_size == @largest").pivot(
        index=["model", "model_key", "model_category"], columns="prediction_type",
        values=["nMSE_mean", "nMSE_std", "nRMSE_mean", "nRMSE_std", "nMAE_mean", "nMAE_std", "macro_R2_mean", "macro_R2_std"],
    )
    id_pivot.columns = [f"{metric}_{prediction}" for metric, prediction in id_pivot.columns]
    id_pivot = id_pivot.reset_index()
    timing = tables["timing_summary"].query("sample_size == @largest").drop(
        columns=["model", "model_category", "sample_size", "fold_count"], errors="ignore"
    )
    full_benchmark = id_pivot.merge(timing, on="model_key", validate="one_to_one")
    full_benchmark = ordered(full_benchmark)
    full_benchmark.to_csv(output_dir / "main_full_size_benchmark.csv", index=False)

    projection_fold = tables["id_fold"].query("sample_size == @largest").pivot(index=["model", "model_key", "model_category", "outer_fold"], columns="prediction_type", values="nRMSE").reset_index()
    projection_fold["delta_nRMSE"] = projection_fold["projected"] - projection_fold["raw"]
    projection_delta = projection_fold.groupby(["model", "model_key", "model_category"], sort=False)["delta_nRMSE"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "delta_nRMSE_mean", "std": "delta_nRMSE_std"})
    physical = tables["physical_summary"].query("sample_size == @largest").drop(
        columns=["sample_size", "fold_count"], errors="ignore"
    )
    physical_value_columns = [
        column for column in physical.columns
        if column not in {"model", "model_key", "model_category", "prediction_type"}
    ]
    physical_pivot = physical.pivot(
        index=["model", "model_key", "model_category"],
        columns="prediction_type",
        values=physical_value_columns,
    )
    physical_pivot.columns = [f"{metric}_{prediction}" for metric, prediction in physical_pivot.columns]
    physical_pivot = physical_pivot.reset_index()
    exact_fold_max = tables["physical_fold"].query("sample_size == @largest").groupby(
        ["model", "model_key", "model_category", "prediction_type"], sort=False
    )["conservation_l2_max"].max().unstack("prediction_type").reset_index().rename(
        columns={"raw": "conservation_l2_exact_max_raw", "projected": "conservation_l2_exact_max_projected"}
    )
    projection = full_benchmark.merge(
        projection_delta,
        on=["model", "model_key", "model_category"],
        validate="one_to_one",
    ).merge(
        physical_pivot,
        on=["model", "model_key", "model_category"],
        validate="one_to_one",
    ).merge(
        exact_fold_max,
        on=["model", "model_key", "model_category"],
        validate="one_to_one",
    )
    ordered(projection).to_csv(output_dir / "main_projection_summary.csv", index=False)

    ood = tables["ood"].merge(
        tables["ood_physical"],
        on=["model", "model_key", "model_category", "regime", "prediction_type"],
        suffixes=("", "_physical"),
        validate="one_to_one",
    )
    ordered(ood).to_csv(output_dir / "main_ood_summary.csv", index=False)

    ood_negative_rows = []
    for key in MODEL_ORDER:
        for regime in ("mild", "severe"):
            prediction = pd.read_parquet(
                run_root / "predictions" / "ood" / f"{key}_{regime}.parquet",
                columns=["raw_negative_standardized_l1", "projected_negative_standardized_l1"],
            )
            for kind in ("raw", "projected"):
                values = prediction[f"{kind}_negative_standardized_l1"].to_numpy(float)
                ood_negative_rows.append(
                    {
                        "model": MODEL_LABELS[key],
                        "model_key": key,
                        "model_category": MODEL_CATEGORIES[key],
                        "regime": regime,
                        "prediction_type": kind,
                        "negative_standardized_l1_mean": float(values.mean()),
                        "negative_standardized_l1_q95": float(np.quantile(values, 0.95)),
                        "negative_standardized_l1_max": float(values.max()),
                    }
                )
    ood_negative = ordered(pd.DataFrame(ood_negative_rows))
    ood_negative.to_csv(output_dir / "ood_negative_magnitude_summary.csv", index=False)

    component_full = tables["component"].query("sample_size == @largest")
    component_summary = component_full.groupby(["model", "model_key", "model_category", "component", "unit", "prediction_type"], sort=False).agg(
        **{f"{column}_mean": (column, "mean") for column in ["MSE", "RMSE", "MAE", "bias", "maximum_absolute_error", "R2", "nMSE_component", "nRMSE_component", "nMAE_component", "negative_count", "negative_magnitude_mean", "projection_displacement_mean", "projection_displacement_standardized_mean"]},
        **{f"{column}_std": (column, "std") for column in ["MSE", "RMSE", "MAE", "bias", "maximum_absolute_error", "R2", "nMSE_component", "nRMSE_component", "nMAE_component", "negative_count", "negative_magnitude_mean", "projection_displacement_mean", "projection_displacement_standardized_mean"]},
    ).reset_index()
    ordered(component_summary).to_csv(output_dir / "supplement_component_summary.csv", index=False)

    composite_full = tables["composite"].query("sample_size == @largest")
    composite_summary = composite_full.groupby(["model", "model_key", "model_category", "composite", "unit", "prediction_type"], sort=False).agg(
        **{f"{column}_mean": (column, "mean") for column in ["MSE", "RMSE", "MAE", "bias", "maximum_absolute_error", "R2"]},
        **{f"{column}_std": (column, "std") for column in ["MSE", "RMSE", "MAE", "bias", "maximum_absolute_error", "R2"]},
    ).reset_index()
    ordered(composite_summary).to_csv(output_dir / "supplement_composite_summary.csv", index=False)

    for key in ("pls_regressor", "multitask_elastic_net_regressor", "multitask_lasso_regressor"):
        coefficients = pd.read_csv(run_root / "models" / "interpretability" / f"{key}_standardized_coefficients.csv")
        coefficients.to_csv(output_dir / f"coefficients_{key}.csv", index=False)

    hyperparameter_rows = []
    largest_selection_size = int(tables["id_fold"]["sample_size"].max())
    outer_fold_count = int(tables["id_fold"]["outer_fold"].nunique())
    for path in sorted((run_root / "tuning").glob("*/*.selected.json")):
        study = path.stem.removesuffix(".selected")
        timing_record = load_json(path.with_name(f"{study}.timing.json"))
        hyperparameter_rows.append(
            {
                "model_key": path.parent.name,
                "model": MODEL_LABELS[path.parent.name],
                "study": study,
                "selection_rows": largest_selection_size if study == "full_id" else largest_selection_size * (outer_fold_count - 1) // outer_fold_count,
                "objective": "four-fold inner nMSE" if outer_fold_count == 5 else "inner-fold nMSE",
                "sampler": "seeded TPE",
                "sampler_seed": 42,
                "completed_trials": int(timing_record["completed_trials"]),
                "search_seconds": float(timing_record["search_seconds"]),
                "selected_settings_json": json.dumps(load_json(path), sort_keys=True, separators=(",", ":")),
            }
        )
    hyperparameters = pd.DataFrame(hyperparameter_rows)
    hyperparameters["model_key"] = pd.Categorical(hyperparameters["model_key"], MODEL_ORDER, ordered=True)
    study_order = [f"outer_{fold}" for fold in sorted(tables["id_fold"]["outer_fold"].unique())] + ["full_id"]
    hyperparameters["study"] = pd.Categorical(hyperparameters["study"], study_order, ordered=True)
    hyperparameters = hyperparameters.sort_values(["model_key", "study"])
    hyperparameters.to_csv(output_dir / "selected_hyperparameters.csv", index=False)

    manifest = load_json(run_root / "manifest.json")
    provenance = {
        "run_id": manifest["run_id"],
        "profile": manifest["profile"],
        "started_utc": manifest["started_utc"],
        "completed_utc": manifest["completed_utc"],
        "contract": manifest["contract"],
        "environment": manifest["environment"],
        "matrix_validation": load_json(run_root / "matrices" / "validation.json"),
    }
    (output_dir / "scientific_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    provenance_rows = [
        {"record": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in provenance.items()
    ]
    pd.DataFrame(provenance_rows).to_csv(output_dir / "scientific_provenance.csv", index=False)

    operators = np.load(run_root / "matrices" / "operators.npz")
    composition = pd.DataFrame(
        np.asarray(operators["composition"], dtype=float),
        index=["COD", "TN", "TP", "TSS"],
        columns=STATE_COLUMNS,
    ).reset_index(names="composite")
    composition.to_csv(output_dir / "composition_matrix.csv", index=False)

    scaling_source = tables["id_summary"].query("prediction_type == 'raw'").merge(
        tables["timing_summary"],
        on=["model", "model_key", "model_category", "sample_size"],
        suffixes=("_accuracy", "_timing"),
        validate="one_to_one",
    )
    ordered(scaling_source).to_csv(output_dir / "figure_main_scaling.csv", index=False)
    delta_all = tables["id_summary"].pivot(
        index=["model", "model_key", "model_category", "sample_size"],
        columns="prediction_type",
        values="nRMSE_mean",
    ).reset_index()
    delta_all["delta_nRMSE"] = delta_all["projected"] - delta_all["raw"]
    ordered(delta_all).to_csv(output_dir / "figure_main_projection_delta.csv", index=False)
    ordered(tables["ood"].query("regime == 'severe'")).to_csv(output_dir / "figure_main_severe_ood.csv", index=False)

    return {"largest_sample_size": largest, "dataset_attempts": attempts.to_dict(orient="records")}


TEX_MODEL_LABELS = {**MODEL_LABELS, "knn_regressor": "$k$-NN"}
TEX_GROUPS = {
    "xgboost_regressor": "Boosting",
    "lightgbm_regressor": "Boosting",
    "catboost_regressor": "Boosting",
    "adaboost_regressor": "Boosting",
    "random_forest_regressor": "Randomized tree ensembles",
    "extra_trees_regressor": "Randomized tree ensembles",
    "svr_regressor": "Kernel",
    "knn_regressor": "Instance based",
    "pls_regressor": "Interpretable models",
    "multitask_elastic_net_regressor": "Interpretable models",
    "multitask_lasso_regressor": "Interpretable models",
    "ann_deep_regressor": "Neural networks",
    "tabnet_regressor": "Neural networks",
}


def tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def tex_number(value: float, digits: int = 3) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "--"
    absolute = abs(number)
    if absolute != 0.0 and (absolute < 10 ** (-digits) or absolute >= 10 ** (digits + 2)):
        exponent = int(math.floor(math.log10(absolute)))
        mantissa = number / (10**exponent)
        return f"${mantissa:.{digits - 1}f}\\times10^{{{exponent}}}$"
    return f"{number:.{digits}f}"


def tex_pm(mean: float, standard_deviation: float, digits: int = 3) -> str:
    return f"${tex_number(mean, digits).strip('$')} \\pm {tex_number(standard_deviation, digits).strip('$')}$"


def write_tex_rows(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_latex_fragments(run_root: Path, tables: dict[str, pd.DataFrame], source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(source_dir / "main_dataset_validation.csv").set_index("set")
    dataset_rows = []
    for label in ("In-distribution", "Mild out-of-distribution", "Severe out-of-distribution"):
        row = dataset.loc[label]
        dataset_rows.append(
            f"{label} & {int(row.attempted):,} & {int(row.retained):,} & {100 * row.retention_rate:.1f} & "
            f"{tex_number(row.maximum_conservation_l2)} & {tex_number(row.minimum_component)} \\\\"
        )
    write_tex_rows(output_dir / "main_dataset_validation_rows.tex", dataset_rows)

    benchmark = pd.read_csv(source_dir / "main_full_size_benchmark.csv").set_index("model_key")
    projection = pd.read_csv(source_dir / "main_projection_summary.csv").set_index("model_key")
    benchmark_rows = []
    projection_rows = []
    for key in MODEL_ORDER:
        row = benchmark.loc[key]
        physics = projection.loc[key]
        raw_accuracy = tex_pm(row.nRMSE_mean_raw, row.nRMSE_std_raw)
        projected_accuracy = (
            r"\shortstack[l]{nRMSE " + tex_pm(row.nRMSE_mean_projected, row.nRMSE_std_projected)
            + r"\\max $v_{mc}=" + tex_number(physics.conservation_l2_exact_max_projected).strip("$")
            + r"$; violations " + f"{100 * physics.mass_violation_rate_mean_projected:.2f}/{100 * physics.nonnegative_violation_rate_mean_projected:.2f}\\%}}"
        )
        timing = (
            r"\shortstack[l]{setup " + tex_pm(row.setup_seconds_mean, row.setup_seconds_std) + r" s\\"
            + "raw " + tex_pm(row.raw_latency_ms_per_sample_mean, row.raw_latency_ms_per_sample_std)
            + r"; projection " + tex_pm(row.projection_latency_ms_per_sample_mean, row.projection_latency_ms_per_sample_std)
            + r"\\end-to-end " + tex_pm(row.end_to_end_latency_ms_per_sample_mean, row.end_to_end_latency_ms_per_sample_std)
            + r" ms sample$^{-1}$}"
        )
        benchmark_rows.append(f"{TEX_MODEL_LABELS[key]} & {raw_accuracy} & {projected_accuracy} & {timing} \\\\")

        paired = (
            r"\shortstack[l]{nRMSE "
            + f"{tex_number(physics.nRMSE_mean_raw)} $\\rightarrow$ {tex_number(physics.nRMSE_mean_projected)}; "
            + r"$\Delta$ " + tex_pm(physics.delta_nRMSE_mean, physics.delta_nRMSE_std)
            + r"\\max $v_{mc}$ raw/projected "
            + f"{tex_number(physics.conservation_l2_exact_max_raw)}/{tex_number(physics.conservation_l2_exact_max_projected)}"
            + r"\\raw $v_{nn}^{*}$ / violation "
            + f"{tex_number(physics.negative_standardized_l1_mean_mean_raw)}/{100 * physics.nonnegative_violation_rate_mean_raw:.2f}\\%"
            + r"\\$d_c^*$ / backtracking "
            + f"{tex_number(physics.standardized_displacement_l2_mean_mean_projected)}/{100 * physics.backtracking_rate_mean_projected:.2f}\\%}}"
        )
        projection_rows.append(f"{TEX_GROUPS[key]} & {TEX_MODEL_LABELS[key]} & {paired} \\\\")
    write_tex_rows(output_dir / "main_full_benchmark_rows.tex", benchmark_rows)
    write_tex_rows(output_dir / "main_projection_rows.tex", projection_rows)

    ood = pd.read_csv(source_dir / "main_ood_summary.csv")
    ood_negative = pd.read_csv(source_dir / "ood_negative_magnitude_summary.csv")
    ood_rows = []
    for key in MODEL_ORDER:
        cells = []
        for regime in ("mild", "severe"):
            raw = ood.query("model_key == @key and regime == @regime and prediction_type == 'raw'").iloc[0]
            projected = ood.query("model_key == @key and regime == @regime and prediction_type == 'projected'").iloc[0]
            raw_negative = ood_negative.query("model_key == @key and regime == @regime and prediction_type == 'raw'").iloc[0]
            cells.append(
                r"\shortstack[l]{nRMSE "
                + f"{tex_number(raw.nRMSE)} $\\rightarrow$ {tex_number(projected.nRMSE)}; $\\Delta$ {tex_number(projected.nRMSE - raw.nRMSE)}"
                + r"\\raw max $v_{mc}$ / $v_{nn}^{*}$ "
                + f"{tex_number(raw.conservation_l2_max)}/{tex_number(raw_negative.negative_standardized_l1_mean)}"
                + r"\\projected violations "
                + f"{100 * projected.mass_violation_rate:.2f}/{100 * projected.nonnegative_violation_rate:.2f}\\%}}"
            )
        ood_rows.append(f"{TEX_MODEL_LABELS[key]} & {cells[0]} & {cells[1]} \\\\")
    write_tex_rows(output_dir / "main_ood_rows.tex", ood_rows)

    id_summary = ordered(tables["id_summary"])
    rows = []
    for row in id_summary.itertuples(index=False):
        rows.append(
            f"{TEX_MODEL_LABELS[str(row.model_key)]} & {int(row.sample_size):,} & {str(row.prediction_type).capitalize()} summary & "
            f"{tex_pm(row.nMSE_mean, row.nMSE_std)} & {tex_pm(row.nRMSE_mean, row.nRMSE_std)} & "
            f"{tex_pm(row.nMAE_mean, row.nMAE_std)} & {tex_pm(row.macro_R2_mean, row.macro_R2_std)} & {int(row.sample_size) // 5:,} \\\\"
        )
    write_tex_rows(output_dir / "s_fold_accuracy_rows.tex", rows)

    physical = tables["physical_summary"]
    timing = tables["timing_summary"]
    rows = []
    for key in MODEL_ORDER:
        for size in sorted(physical["sample_size"].unique()):
            raw = physical.query("model_key == @key and sample_size == @size and prediction_type == 'raw'").iloc[0]
            projected = physical.query("model_key == @key and sample_size == @size and prediction_type == 'projected'").iloc[0]
            time = timing.query("model_key == @key and sample_size == @size").iloc[0]
            rows.append(
                f"{TEX_MODEL_LABELS[key]} & {int(size):,} & Summary & "
                f"{tex_number(raw.conservation_l2_mean_mean)} $\\rightarrow$ {tex_number(projected.conservation_l2_mean_mean)} & "
                f"{100 * raw.nonnegative_violation_rate_mean:.2f} $\\rightarrow$ {100 * projected.nonnegative_violation_rate_mean:.2f} & "
                f"{tex_pm(projected.standardized_displacement_l2_mean_mean, projected.standardized_displacement_l2_mean_std)} & "
                f"{100 * projected.backtracking_rate_mean:.2f} & {tex_pm(time.setup_seconds_mean, time.setup_seconds_std)} & "
                f"{tex_pm(time.raw_latency_ms_per_sample_mean, time.raw_latency_ms_per_sample_std)} & "
                f"{tex_pm(time.projection_latency_ms_per_sample_mean, time.projection_latency_ms_per_sample_std)} \\\\"
            )
    write_tex_rows(output_dir / "s_fold_physics_timing_rows.tex", rows)

    component_summary = ordered(pd.read_csv(source_dir / "supplement_component_summary.csv"))
    rows = []
    for row in component_summary.itertuples(index=False):
        displacement = tex_pm(row.projection_displacement_standardized_mean_mean, row.projection_displacement_standardized_mean_std) if str(row.prediction_type) == "projected" else "--"
        rows.append(
            f"{tex_escape(row.component)} & {TEX_MODEL_LABELS[str(row.model_key)]} & {str(row.prediction_type).capitalize()} & Summary & "
            f"{tex_pm(row.MSE_mean, row.MSE_std)} & {tex_pm(row.RMSE_mean, row.RMSE_std)} & {tex_pm(row.MAE_mean, row.MAE_std)} & "
            f"{tex_pm(row.bias_mean, row.bias_std)} & {tex_pm(row.maximum_absolute_error_mean, row.maximum_absolute_error_std)} & "
            f"{tex_pm(row.nMSE_component_mean, row.nMSE_component_std)} & {tex_pm(row.nRMSE_component_mean, row.nRMSE_component_std)} & "
            f"{tex_pm(row.nMAE_component_mean, row.nMAE_component_std)} & {displacement} \\\\"
        )
    write_tex_rows(output_dir / "s_component_results_rows.tex", rows)

    full_size = int(tables["component"]["sample_size"].max())
    component_full = tables["component"].query("sample_size == @full_size")
    rows = []
    for key in MODEL_ORDER:
        for component in STATE_COLUMNS:
            raw = component_full.query("model_key == @key and component == @component and prediction_type == 'raw'")
            projected = component_full.query("model_key == @key and component == @component and prediction_type == 'projected'")
            rows.append(
                f"{tex_escape(component)} & {TEX_MODEL_LABELS[key]} & Summary & {int(raw.negative_count.sum()):,} & "
                f"{tex_number(raw.negative_magnitude_mean.mean())} & {tex_number(projected.projection_displacement_mean.mean())} & "
                f"{tex_number(projected.projection_displacement_standardized_mean.mean())} \\\\"
            )
    write_tex_rows(output_dir / "s_component_physics_rows.tex", rows)

    composite_summary = ordered(pd.read_csv(source_dir / "supplement_composite_summary.csv"))
    rows = []
    for row in composite_summary.itertuples(index=False):
        rows.append(
            f"{tex_escape(row.composite)} & {TEX_MODEL_LABELS[str(row.model_key)]} & {str(row.prediction_type).capitalize()} & Summary & "
            f"{tex_pm(row.MSE_mean, row.MSE_std)} & {tex_pm(row.RMSE_mean, row.RMSE_std)} & {tex_pm(row.MAE_mean, row.MAE_std)} & "
            f"{tex_pm(row.bias_mean, row.bias_std)} & {tex_pm(row.R2_mean, row.R2_std)} \\\\"
        )
    write_tex_rows(output_dir / "s_composite_results_rows.tex", rows)

    composition = pd.read_csv(source_dir / "composition_matrix.csv")
    composition_rows = []
    for _, row in composition.iterrows():
        coefficients = "; ".join(
            f"{name}={float(row[name]):.6g}" for name in STATE_COLUMNS if float(row[name]) != 0.0
        )
        composition_rows.append(
            f"{tex_escape(row['composite'])} & {tex_escape(coefficients)} \\\\"
        )
    write_tex_rows(output_dir / "s_composition_matrix_rows.tex", composition_rows)

    coefficient_rows = []
    for key in ("pls_regressor", "multitask_elastic_net_regressor", "multitask_lasso_regressor"):
        coefficients = pd.read_csv(source_dir / f"coefficients_{key}.csv")
        for _, row in coefficients.iterrows():
            vector = "; ".join(f"{name}={float(row[name]):.4g}" for name in STATE_COLUMNS)
            coefficient_rows.append(
                f"{TEX_MODEL_LABELS[key]} & {tex_escape(row['feature'])} & {tex_escape(vector)} & {tex_number(row['row_l2_norm'])} & "
                f"{'Yes' if bool(row['active_at_1e-12']) else 'No'} \\\\"
            )
    write_tex_rows(output_dir / "s_interpretable_coefficients_rows.tex", coefficient_rows)

    acceptance_rows = []
    for regime, label in (("Mild out-of-distribution", "Mild"), ("Severe out-of-distribution", "Severe")):
        row = dataset.loc[regime]
        acceptance_rows.append(
            f"{label} & {int(row.attempted):,} & {int(row.solver_accepted):,} & {int(row.solver_failed):,} & "
            f"{int(row.valid_unselected):,} & {int(row.retained):,} & {100 * row.solver_acceptance_rate:.1f} & "
            f"{100 * row.retention_rate:.1f} & {tex_number(row.maximum_conservation_l2)} \\\\"
        )
    write_tex_rows(output_dir / "s_ood_acceptance_rows.tex", acceptance_rows)

    ood_rows = []
    for row in ordered(ood).itertuples(index=False):
        negative = ood_negative.query(
            "model_key == @row.model_key and regime == @row.regime and prediction_type == @row.prediction_type"
        ).iloc[0]
        ood_rows.append(
            f"{str(row.regime).capitalize()} & {TEX_MODEL_LABELS[str(row.model_key)]} & {str(row.prediction_type).capitalize()} & "
            f"{tex_number(row.nMSE)} & {tex_number(row.nRMSE)} & {tex_number(row.nMAE)} & {tex_number(row.macro_R2)} & "
            f"{tex_number(row.conservation_l2_max)} & {100 * row.nonnegative_violation_rate:.2f} & {tex_number(row.standardized_displacement_l2_mean)} \\\\"
        )
    write_tex_rows(output_dir / "s_ood_results_rows.tex", ood_rows)

    hyperparameters = pd.read_csv(source_dir / "selected_hyperparameters.csv")
    hyperparameter_rows = []
    for row in hyperparameters.itertuples(index=False):
        fold = "Full ID" if row.study == "full_id" else row.study.replace("outer_", "")
        scale = f"{int(row.selection_rows):,} selection rows"
        settings_for_tex = json.dumps(json.loads(str(row.selected_settings_json)), sort_keys=True)
        hyperparameter_rows.append(
            f"{TEX_MODEL_LABELS[str(row.model_key)]} & {tex_escape(fold)} & {scale} & "
            f"{tex_escape(row.objective)}; {tex_escape(row.sampler)}, seed {int(row.sampler_seed)}; "
            f"search {tex_number(row.search_seconds)} s ({int(row.completed_trials)} trials); {tex_escape(settings_for_tex)} \\\\"
        )
    write_tex_rows(output_dir / "s_hyperparameters_rows.tex", hyperparameter_rows)

    timing_rows = []
    for row in ordered(tables["timing_summary"]).itertuples(index=False):
        timing_rows.append(
            f"{TEX_MODEL_LABELS[str(row.model_key)]} & {int(row.sample_size):,} & Summary & "
            f"{tex_pm(row.setup_seconds_mean, row.setup_seconds_std)} & {tex_pm(row.raw_latency_ms_per_sample_mean, row.raw_latency_ms_per_sample_std)} & "
            f"{tex_pm(row.projection_latency_ms_per_sample_mean, row.projection_latency_ms_per_sample_std)} & "
            f"{tex_pm(row.end_to_end_latency_ms_per_sample_mean, row.end_to_end_latency_ms_per_sample_std)} \\\\"
        )
    write_tex_rows(output_dir / "s_timings_rows.tex", timing_rows)

    manifest = load_json(run_root / "manifest.json")
    environment = manifest["environment"]
    packages = environment["packages"]
    resolved = load_json(run_root / "inputs" / "params.resolved.json")
    general_packages = [
        f"scikit-learn {packages['scikit-learn']}",
        f"XGBoost {packages['xgboost']}",
        f"LightGBM {packages['lightgbm']}",
        f"CatBoost {packages['catboost']}",
        f"NumPy {packages['numpy']}",
        f"SciPy {packages['scipy']}",
        f"pandas {packages['pandas']}",
        f"Optuna {packages['optuna']}",
        f"Matplotlib {packages['matplotlib']}",
        f"PyArrow {packages['pyarrow']}",
    ]
    row_end = " " + chr(92) * 2
    provenance_rows = [
        "Tree, kernel, neighbor, linear, and MLP package versions & " + tex_escape("; ".join(general_packages)) + row_end,
        "TabNet and PyTorch versions & " + tex_escape(f"pytorch-tabnet {packages['pytorch-tabnet']}; PyTorch {packages['torch']}") + row_end,
        "TabNet device and internal precision & CPU; single precision" + row_end,
        "Processor and installed memory & " + tex_escape(f"{environment['cpu']}; {environment['ram_bytes'] / 2**30:.1f} GiB installed RAM") + row_end,
        "Operating system and thread limits & " + tex_escape(f"{environment['platform']}; {environment['logical_cores']} logical cores; benchmark thread limit {resolved['run']['threads']}") + row_end,
        "Numeric precision by computation stage & Mechanistic states, matrices, preprocessing, returned predictions, projection, and metrics: double precision; TabNet internal tensors: single precision" + row_end,
    ]
    write_tex_rows(output_dir / "s_provenance_rows.tex", provenance_rows)


def derive_descriptive_findings(
    tables: dict[str, pd.DataFrame],
    source_dir: Path,
) -> dict[str, Any]:
    largest = int(tables["id_summary"]["sample_size"].max())
    smallest = int(tables["id_summary"]["sample_size"].min())
    full = pd.read_csv(source_dir / "main_full_size_benchmark.csv").set_index("model_key")
    projection = pd.read_csv(source_dir / "main_projection_summary.csv").set_index("model_key")
    full_rows = []
    for key in MODEL_ORDER:
        row = full.loc[key]
        physics = projection.loc[key]
        full_rows.append(
            {
                "model_key": key,
                "model": MODEL_LABELS[key],
                "raw_nRMSE_mean": float(row.nRMSE_mean_raw),
                "raw_nRMSE_sd": float(row.nRMSE_std_raw),
                "projected_nRMSE_mean": float(row.nRMSE_mean_projected),
                "projected_nRMSE_sd": float(row.nRMSE_std_projected),
                "paired_delta_nRMSE_mean": float(physics.delta_nRMSE_mean),
                "paired_delta_nRMSE_sd": float(physics.delta_nRMSE_std),
                "setup_seconds_mean": float(row.setup_seconds_mean),
                "raw_latency_ms_per_sample_mean": float(row.raw_latency_ms_per_sample_mean),
                "projection_latency_ms_per_sample_mean": float(row.projection_latency_ms_per_sample_mean),
                "end_to_end_latency_ms_per_sample_mean": float(row.end_to_end_latency_ms_per_sample_mean),
                "raw_mass_violation_rate_mean": float(physics.mass_violation_rate_mean_raw),
                "raw_nonnegative_violation_rate_mean": float(physics.nonnegative_violation_rate_mean_raw),
                "projected_mass_violation_rate_mean": float(physics.mass_violation_rate_mean_projected),
                "projected_nonnegative_violation_rate_mean": float(physics.nonnegative_violation_rate_mean_projected),
                "standardized_displacement_mean": float(physics.standardized_displacement_l2_mean_mean_projected),
                "backtracking_rate_mean": float(physics.backtracking_rate_mean_projected),
            }
        )
    raw_best = min(full_rows, key=lambda row: row["raw_nRMSE_mean"])
    projected_best = min(full_rows, key=lambda row: row["projected_nRMSE_mean"])

    delta = tables["id_summary"].pivot(
        index=["model_key", "sample_size"], columns="prediction_type", values="nRMSE_mean"
    ).reset_index()
    delta["delta"] = delta["projected"] - delta["raw"]
    tolerance = 1e-12
    projection_counts = {
        "lower_nRMSE_cells": int((delta["delta"] < -tolerance).sum()),
        "higher_nRMSE_cells": int((delta["delta"] > tolerance).sum()),
        "unchanged_nRMSE_cells": int((delta["delta"].abs() <= tolerance).sum()),
        "total_model_size_cells": int(len(delta)),
        "minimum_delta_nRMSE": float(delta["delta"].min()),
        "maximum_delta_nRMSE": float(delta["delta"].max()),
    }

    learning_rows = []
    raw_summary = tables["id_summary"].query("prediction_type == 'raw'")
    for key in MODEL_ORDER:
        start = float(raw_summary.query("model_key == @key and sample_size == @smallest")["nRMSE_mean"].iloc[0])
        end = float(raw_summary.query("model_key == @key and sample_size == @largest")["nRMSE_mean"].iloc[0])
        learning_rows.append(
            {
                "model_key": key,
                "model": MODEL_LABELS[key],
                "smallest_nRMSE": start,
                "largest_nRMSE": end,
                "relative_change": (end - start) / start,
            }
        )

    ood = pd.read_csv(source_dir / "main_ood_summary.csv")
    ood_rows = []
    for key in MODEL_ORDER:
        model_rows: dict[str, Any] = {"model_key": key, "model": MODEL_LABELS[key]}
        for regime in ("mild", "severe"):
            raw = ood.query("model_key == @key and regime == @regime and prediction_type == 'raw'").iloc[0]
            projected = ood.query("model_key == @key and regime == @regime and prediction_type == 'projected'").iloc[0]
            model_rows[regime] = {
                "raw_nRMSE": float(raw.nRMSE),
                "projected_nRMSE": float(projected.nRMSE),
                "delta_nRMSE": float(projected.nRMSE - raw.nRMSE),
                "raw_mass_violation_rate": float(raw.mass_violation_rate),
                "raw_nonnegative_violation_rate": float(raw.nonnegative_violation_rate),
                "projected_mass_violation_rate": float(projected.mass_violation_rate),
                "projected_nonnegative_violation_rate": float(projected.nonnegative_violation_rate),
            }
        model_rows["raw_severe_minus_mild_nRMSE"] = model_rows["severe"]["raw_nRMSE"] - model_rows["mild"]["raw_nRMSE"]
        ood_rows.append(model_rows)

    coefficient_support = []
    for key in ("pls_regressor", "multitask_elastic_net_regressor", "multitask_lasso_regressor"):
        coefficients = pd.read_csv(source_dir / f"coefficients_{key}.csv")
        ordered_coefficients = coefficients.sort_values("row_l2_norm", ascending=False)
        coefficient_support.append(
            {
                "model_key": key,
                "model": MODEL_LABELS[key],
                "active_features_at_1e-12": int(coefficients["active_at_1e-12"].astype(bool).sum()),
                "top_five_features_by_row_l2_norm": ordered_coefficients["feature"].head(5).tolist(),
            }
        )

    return {
        "smallest_sample_size": smallest,
        "largest_sample_size": largest,
        "full_size_models": full_rows,
        "lowest_observed_raw_mean_nRMSE": raw_best,
        "lowest_observed_projected_mean_nRMSE": projected_best,
        "projection_effect_across_model_size_cells": projection_counts,
        "learning_curve_endpoints": learning_rows,
        "ood_models": ood_rows,
        "interpretable_model_coefficient_support": coefficient_support,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("results/article_final_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/wip/generated/article_final_v2"))
    parser.add_argument("--source-root", type=Path, default=Path("."), help="Repository root containing the accepted notebook, config, lock, and paths file.")
    parser.add_argument("--allow-ineligible", action="store_true", help="Development only: permit a completed smoke bundle.")
    parser.add_argument("--skip-inventory-hashes", action="store_true", help="Skip per-artifact SHA-256 verification; manifest hash is still checked.")
    parser.add_argument("--allow-existing-output", action="store_true", help="Development only: permit overwriting named files in an existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    final_output_dir = args.output_dir.resolve()
    output_dir = final_output_dir
    source_root = args.source_root.resolve()
    require(not final_output_dir.is_relative_to(run_root), "Derived paper assets may not be written inside the immutable run root.")
    if not args.allow_ineligible:
        require(not args.skip_inventory_hashes, "Article generation may not skip artifact hash verification.")
        require(not args.allow_existing_output, "Article generation requires a new, clean output directory.")
        require(not final_output_dir.exists(), f"Article output path already exists: {final_output_dir}")
    if final_output_dir.exists() and any(final_output_dir.iterdir()):
        require(args.allow_ineligible and args.allow_existing_output, f"Output directory is not empty: {final_output_dir}")
    if not args.allow_ineligible:
        final_output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir = final_output_dir.parent / f".{final_output_dir.name}.staging-{os.getpid()}"
        require(not output_dir.exists(), f"Staging directory already exists: {output_dir}")
    manifest = verify_bundle(run_root, allow_ineligible=args.allow_ineligible, verify_inventory=not args.skip_inventory_hashes)
    strict_full = manifest.get("profile") == "full"
    source_contract = verify_source_contract(run_root, manifest, source_root) if strict_full else {}
    tables = load_and_validate_tables(run_root, strict_full=strict_full)
    resolved = load_json(run_root / "inputs" / "params.resolved.json")
    scientific_contract = validate_scientific_contract(
        run_root,
        manifest,
        resolved,
        tables,
        strict_full=strict_full,
    )
    dataset_contract = validate_datasets_and_splits(run_root, resolved, strict_full=strict_full)
    timing_contract = validate_timing_repetitions(
        run_root,
        resolved,
        tables,
        strict_full=strict_full,
    )
    tau = float(resolved["projection"]["tau"])
    physical_contract = prediction_contract(run_root, tau, tables, strict_full=strict_full)
    table_summary = export_analysis_tables(run_root, tables, output_dir / "source_data")
    pd.DataFrame(
        [
            {"regime": regime, **values}
            for regime, values in dataset_contract["ood_normalized_exterior_distance"].items()
        ]
    ).to_csv(output_dir / "source_data" / "ood_design_distance_summary.csv", index=False)
    build_latex_fragments(
        run_root,
        tables,
        output_dir / "source_data",
        output_dir / "tables",
    )
    descriptive_findings = derive_descriptive_findings(tables, output_dir / "source_data")
    (output_dir / "descriptive_findings.json").write_text(
        json.dumps(descriptive_findings, indent=2) + "\n",
        encoding="utf-8",
    )
    style()
    build_main_figures(tables, output_dir / "figures")
    build_supplement_figures(run_root, tables, output_dir / "figures")
    summary = {
        "run_id": manifest["run_id"],
        "profile": manifest["profile"],
        "manifest_sha256": sha256_file(run_root / "manifest.json"),
        "finalizer_sha256": sha256_file(Path(__file__).resolve()),
        "source_contract": source_contract,
        "scientific_contract": scientific_contract,
        "dataset_contract": dataset_contract,
        "timing_contract": timing_contract,
        "physical_contract": physical_contract,
        "descriptive_findings_path": "descriptive_findings.json",
        **table_summary,
        "generation_options": {
            "allow_ineligible": bool(args.allow_ineligible),
            "inventory_hashes_verified": not bool(args.skip_inventory_hashes),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "finalizer_snapshot.py").write_bytes(Path(__file__).resolve().read_bytes())
    (output_dir / "verified_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    generated_assets = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "generation_manifest.json":
            generated_assets.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    generation_manifest = {
        "source_run_id": manifest["run_id"],
        "source_manifest_sha256": sha256_file(run_root / "manifest.json"),
        "finalizer_sha256": sha256_file(Path(__file__).resolve()),
        "assets": generated_assets,
    }
    (output_dir / "generation_manifest.json").write_text(json.dumps(generation_manifest, indent=2) + "\n", encoding="utf-8")
    if output_dir != final_output_dir:
        require(not final_output_dir.exists(), f"Final output appeared during generation: {final_output_dir}")
        output_dir.replace(final_output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
