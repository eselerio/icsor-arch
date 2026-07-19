"""Load trusted surrogate bundles serialized from ``main.ipynb``.

The notebook serializes ``FittedSurrogate`` and ``TabNetRegressorAdapter``
instances while the classes live in ``__main__``.  A normal fresh Python
process therefore cannot resolve those two pickle globals.  This module
temporarily registers compatible, inference-only definitions in ``__main__``
while joblib unpickles a bundle, then restores the prior namespace.

Joblib bundles are pickle-based and can execute code while loading.  Use this
utility only for trusted run artifacts.  Supplying ``expected_sha256`` binds a
load to a previously recorded artifact digest.  The CLI smoke check requires a
completed run, verifies its manifest and all model hashes, loads the exact
13-model roster, and compares fresh predictions with the stored OOD rows.
It never writes to the run directory.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd


MODEL_KEYS = (
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
)


def _positive_worker_count(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)) and int(value) > 0:
        return int(value)
    return None


def _infer_worker_count(estimator: Any) -> int:
    """Recover the explicit worker count persisted by the notebook."""

    direct = _positive_worker_count(getattr(estimator, "n_jobs", None))
    if direct is not None:
        return direct
    nested = getattr(estimator, "estimator", None)
    nested_count = _positive_worker_count(getattr(nested, "n_jobs", None))
    return nested_count if nested_count is not None else 1


class TabNetRegressorAdapter:
    """Inference-compatible definition for the notebook's TabNet adapter."""

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.model: Any | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("The loaded TabNet adapter does not contain a fitted model.")
        values = np.asarray(X, dtype=np.float32)
        return np.asarray(self.model.predict(values), dtype=np.float64)


@dataclass
class FittedSurrogate:
    """Inference-compatible definition for the notebook's fitted bundle."""

    name: str
    estimator: Any
    feature_scaler: Any | None
    target_scaler: Any | None

    def predict(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"Expected a two-dimensional feature matrix; got {values.shape}.")
        if self.feature_scaler is not None:
            values = self.feature_scaler.transform(values)
        workers = int(getattr(self, "_compat_workers", _infer_worker_count(self.estimator)))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
                category=UserWarning,
            )
            with joblib.parallel_backend("threading", n_jobs=workers):
                predicted = np.asarray(self.estimator.predict(values), dtype=np.float64)
        if predicted.ndim == 1:
            predicted = predicted[:, None]
        if self.target_scaler is not None:
            predicted = self.target_scaler.inverse_transform(predicted)
        predicted = np.asarray(predicted, dtype=np.float64)
        expected_targets = int(getattr(self, "_compat_expected_targets", 20))
        expected_shape = (len(values), expected_targets)
        if predicted.shape != expected_shape or not np.all(np.isfinite(predicted)):
            raise RuntimeError(
                f"{self.name} returned {predicted.shape}; expected finite values with shape {expected_shape}."
            )
        return predicted


@contextlib.contextmanager
def _notebook_main_class_registry() -> Iterator[None]:
    """Temporarily expose compatibility classes under their pickle globals."""

    main_module = sys.modules["__main__"]
    sentinel = object()
    prior = {
        "FittedSurrogate": getattr(main_module, "FittedSurrogate", sentinel),
        "TabNetRegressorAdapter": getattr(main_module, "TabNetRegressorAdapter", sentinel),
    }
    setattr(main_module, "FittedSurrogate", FittedSurrogate)
    setattr(main_module, "TabNetRegressorAdapter", TabNetRegressorAdapter)
    try:
        yield
    finally:
        for name, value in prior.items():
            if value is sentinel:
                with contextlib.suppress(AttributeError):
                    delattr(main_module, name)
            else:
                setattr(main_module, name, value)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_surrogate_bundle(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_targets: int = 20,
    workers: int | None = None,
) -> FittedSurrogate:
    """Load one trusted notebook bundle for inference in a fresh process.

    Parameters
    ----------
    path:
        Path to one ``models/*.joblib`` artifact produced by ``main.ipynb``.
    expected_sha256:
        Optional lowercase or uppercase SHA-256 digest recorded by the run
        manifest.  Loading is refused when the digest differs.
    expected_targets:
        Required output width; the article benchmark uses 20.
    workers:
        Optional positive joblib thread count.  When omitted, the loader uses
        the positive ``n_jobs`` value stored on the estimator and otherwise 1.

    Notes
    -----
    This function restores class-name compatibility; it does not make pickle
    safe.  Never load an untrusted joblib file.
    """

    bundle_path = Path(path).expanduser().resolve()
    if bundle_path.suffix.lower() != ".joblib" or not bundle_path.is_file():
        raise ValueError(f"Expected an existing .joblib bundle, got {bundle_path}.")
    if expected_targets <= 0:
        raise ValueError("expected_targets must be positive.")
    if workers is not None and workers <= 0:
        raise ValueError("workers must be positive when supplied.")
    if expected_sha256 is not None:
        expected_digest = str(expected_sha256).strip().lower()
        if len(expected_digest) != 64 or any(character not in "0123456789abcdef" for character in expected_digest):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest.")
        observed_digest = sha256_file(bundle_path)
        if observed_digest != expected_digest:
            raise RuntimeError(
                f"Bundle hash mismatch for {bundle_path}: expected {expected_digest}, observed {observed_digest}."
            )

    with _notebook_main_class_registry():
        loaded = joblib.load(bundle_path)
    if not isinstance(loaded, FittedSurrogate):
        raise TypeError(f"Unexpected bundle type {type(loaded)!r} in {bundle_path}.")
    if not isinstance(loaded.name, str) or not loaded.name:
        raise RuntimeError(f"The bundle in {bundle_path} has no model key.")
    loaded._compat_expected_targets = int(expected_targets)
    loaded._compat_workers = int(workers) if workers is not None else _infer_worker_count(loaded.estimator)
    if workers is not None and hasattr(loaded.estimator, "n_jobs"):
        loaded.estimator.n_jobs = int(workers)
    return loaded


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _verified_completed_manifest(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "manifest.json"
    completed_path = run_root / "COMPLETED.json"
    digest_path = run_root / "manifest.sha256"
    for required in (manifest_path, completed_path, digest_path):
        if not required.is_file():
            raise FileNotFoundError(f"Completed-run artifact is missing: {required}.")
    manifest_digest = sha256_file(manifest_path)
    manifest = _read_json(manifest_path)
    completed = _read_json(completed_path)
    recorded_digest = digest_path.read_text(encoding="ascii").split()[0].lower()
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Run manifest is not complete: {run_root}.")
    if completed.get("manifest_sha256") != manifest_digest or recorded_digest != manifest_digest:
        raise RuntimeError(f"Completed-run manifest digest mismatch: {run_root}.")
    if completed.get("run_id") != manifest.get("run_id"):
        raise RuntimeError(f"COMPLETED.json and manifest.json disagree on run_id in {run_root}.")
    return manifest


def _verified_inventory_path(
    run_root: Path,
    inventory: dict[str, Any],
    relative_path: str,
) -> Path:
    entry = inventory.get(relative_path)
    if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
        raise RuntimeError(f"Manifest inventory lacks a digest for {relative_path}.")
    path = run_root / Path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest-inventoried file is missing: {path}.")
    expected_bytes = entry.get("bytes")
    if isinstance(expected_bytes, int) and path.stat().st_size != expected_bytes:
        raise RuntimeError(f"Manifest-inventoried byte count differs for {relative_path}.")
    observed_digest = sha256_file(path)
    if observed_digest != str(entry["sha256"]).lower():
        raise RuntimeError(f"Manifest-inventoried SHA-256 differs for {relative_path}.")
    return path


def smoke_check_run(
    run_root: str | Path,
    *,
    regime: str = "mild",
    tolerance: float = 1e-10,
    workers: int | None = None,
) -> dict[str, Any]:
    """Read-only load and prediction check for a completed 13-model run."""

    root = Path(run_root).expanduser().resolve()
    if regime not in {"mild", "severe"}:
        raise ValueError("regime must be 'mild' or 'severe'.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")
    if workers is not None and workers <= 0:
        raise ValueError("workers must be positive when supplied.")
    manifest = _verified_completed_manifest(root)
    inventory = manifest.get("artifact_inventory")
    if not isinstance(inventory, dict):
        raise RuntimeError("Completed manifest does not contain an artifact inventory.")

    model_dir = root / "models"
    observed_keys = {path.stem for path in model_dir.glob("*.joblib") if path.is_file()}
    expected_keys = set(MODEL_KEYS)
    if observed_keys != expected_keys:
        raise RuntimeError(
            f"Model roster mismatch: missing={sorted(expected_keys - observed_keys)}, "
            f"unexpected={sorted(observed_keys - expected_keys)}."
        )

    resolved_path = _verified_inventory_path(root, inventory, "inputs/params.resolved.json")
    resolved = _read_json(resolved_path)
    workbook = resolved["simulation"]["workbook"]
    state_columns = list(workbook["state_columns"])
    operational_columns = list(resolved["simulation"]["operational_columns"])
    if len(state_columns) != 20:
        raise RuntimeError(f"Expected 20 state columns; found {len(state_columns)}.")
    feature_columns = [*operational_columns, *(f"In_{name}" for name in state_columns)]
    raw_columns = [f"raw_{name}" for name in state_columns]

    dataset_relative = f"datasets/ood_{regime}.parquet"
    dataset = pd.read_parquet(_verified_inventory_path(root, inventory, dataset_relative))
    if dataset["sample_id"].astype(str).duplicated().any():
        raise RuntimeError(f"OOD {regime} sample identifiers are not unique.")
    dataset = dataset.assign(sample_id=dataset["sample_id"].astype(str)).set_index("sample_id", drop=False)

    model_results: list[dict[str, Any]] = []
    global_maximum = 0.0
    for model_key in MODEL_KEYS:
        relative_bundle = f"models/{model_key}.joblib"
        inventory_entry = inventory.get(relative_bundle)
        if not isinstance(inventory_entry, dict) or not isinstance(inventory_entry.get("sha256"), str):
            raise RuntimeError(f"Manifest inventory lacks a digest for {relative_bundle}.")
        prediction_relative = f"predictions/ood/{model_key}_{regime}.parquet"
        prediction_path = _verified_inventory_path(root, inventory, prediction_relative)
        stored = pd.read_parquet(prediction_path)
        sample_ids = stored["sample_id"].astype(str)
        if sample_ids.duplicated().any():
            raise RuntimeError(f"Stored predictions contain duplicate sample IDs for {model_key}.")
        missing_ids = sorted(set(sample_ids) - set(dataset.index))
        if missing_ids:
            raise RuntimeError(f"Stored predictions contain unknown sample IDs for {model_key}: {missing_ids[:3]}.")
        if len(sample_ids) != len(dataset) or set(sample_ids) != set(dataset.index):
            raise RuntimeError(f"Stored predictions do not cover the complete OOD {regime} dataset for {model_key}.")
        aligned = dataset.loc[sample_ids]
        X = aligned[feature_columns].to_numpy(np.float64)
        expected = stored[raw_columns].to_numpy(np.float64)
        fitted = load_surrogate_bundle(
            root / relative_bundle,
            expected_sha256=inventory_entry["sha256"],
            expected_targets=len(state_columns),
            workers=workers,
        )
        if fitted.name != model_key:
            raise RuntimeError(f"Bundle {relative_bundle} identifies itself as {fitted.name!r}.")
        observed = fitted.predict(X)
        maximum_difference = float(np.max(np.abs(observed - expected)))
        global_maximum = max(global_maximum, maximum_difference)
        model_results.append(
            {
                "model_key": model_key,
                "samples": int(len(observed)),
                "targets": int(observed.shape[1]),
                "workers": int(fitted._compat_workers),
                "maximum_absolute_difference": maximum_difference,
                "within_tolerance": bool(maximum_difference <= tolerance),
            }
        )

    result = {
        "status": "passed" if all(row["within_tolerance"] for row in model_results) else "failed",
        "run_id": manifest.get("run_id"),
        "run_root": str(root),
        "regime": regime,
        "model_count": len(model_results),
        "samples_per_model": int(len(dataset)),
        "targets_per_sample": len(state_columns),
        "tolerance": float(tolerance),
        "maximum_absolute_difference": global_maximum,
        "models": model_results,
    }
    if result["status"] != "passed":
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-check-run-root",
        required=True,
        type=Path,
        help="Completed run root whose 13 model bundles and stored OOD predictions will be checked read-only.",
    )
    parser.add_argument("--regime", choices=("mild", "severe"), default="mild")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional positive inference thread count; default recovers each bundle's persisted n_jobs or uses 1.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = smoke_check_run(
            args.smoke_check_run_root,
            regime=args.regime,
            tolerance=args.tolerance,
            workers=args.workers,
        )
    except Exception as error:
        print(
            json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)}, indent=2),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
