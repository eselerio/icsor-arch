"""Apply the authorized, TabNet-only protocol amendment to article_final_v2."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (ROOT / "results" / "article_final_v2").resolve()
BACKUP_ROOT = (ROOT / "results" / "recovery_backups").resolve()
OLD_CONFIG_SHA256 = "9b8ec5ec3ba6be6e16c95137fc89c75513a77180d0352f004435d2914c52b6c4"
EXPECTED_NOTEBOOK_SHA256 = "b7471b8c458b3af9ceca050fedbbb732869c8d2168e98caba9991db5a37856b2"
EXPECTED_TABNET_FILES = {"outer_0.sampler.pkl", "outer_0.sqlite3", "outer_0.timing.json"}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def json_payload(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode()


def require_within(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if resolved != parent and parent not in resolved.parents:
        raise RuntimeError(f"Refusing path outside {parent}: {resolved}")


def main() -> None:
    manifest_path = RUN_ROOT / "manifest.json"
    resolved_path = RUN_ROOT / "inputs" / "params.resolved.json"
    manifest = load_json(manifest_path)
    old_resolved = load_json(resolved_path)
    config = load_json(ROOT / "config" / "params.json")

    if manifest.get("status") != "running" or manifest.get("profile") != "full":
        raise RuntimeError("The target must be an incomplete full-profile run.")
    contract = manifest.get("contract", {})
    if contract.get("config_sha256") != OLD_CONFIG_SHA256:
        raise RuntimeError("The run is not on the expected pre-amendment configuration.")
    if contract.get("notebook_sha256") != EXPECTED_NOTEBOOK_SHA256:
        raise RuntimeError("The notebook contract differs from the reviewed source.")
    if sha256_json(old_resolved) != OLD_CONFIG_SHA256:
        raise RuntimeError("The pre-amendment resolved configuration hash does not reproduce.")
    if sha256_file(ROOT / "main.ipynb") != EXPECTED_NOTEBOOK_SHA256:
        raise RuntimeError("main.ipynb changed while preparing the scoped migration.")

    active = json.loads(json.dumps(config))
    active["run"]["profile"] = "full"
    new_config_sha256 = sha256_json(active)
    if new_config_sha256 == OLD_CONFIG_SHA256:
        raise RuntimeError("The accepted configuration did not change.")
    old_models = old_resolved["evaluation"]["models"]
    new_models = active["evaluation"]["models"]
    changed_models = [name for name in old_models if old_models[name] != new_models[name]]
    if changed_models != ["tabnet_regressor"]:
        raise RuntimeError(f"The migration is not TabNet-only: {changed_models}")

    tabnet_dir = RUN_ROOT / "tuning" / "tabnet_regressor"
    require_within(tabnet_dir, RUN_ROOT)
    observed = {path.name for path in tabnet_dir.iterdir()} if tabnet_dir.is_dir() else set()
    if observed != EXPECTED_TABNET_FILES:
        raise RuntimeError(f"Unexpected exploratory TabNet artifact set: {sorted(observed)}")
    unexpected = [
        path.relative_to(RUN_ROOT).as_posix()
        for path in RUN_ROOT.rglob("*")
        if path.is_file()
        and ("tabnet_regressor" in path.name or "tabnet_regressor" in path.parts)
        and tabnet_dir not in path.parents
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected TabNet-derived artifacts exist: {unexpected}")

    timestamp = datetime.now(timezone.utc)
    backup_dir = BACKUP_ROOT / f"tabnet_protocol_migration_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    require_within(backup_dir, BACKUP_ROOT)
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, backup_dir / "manifest.before.json")
    shutil.copy2(resolved_path, backup_dir / "params.resolved.before.json")
    archived = backup_dir / "invalidated_tabnet_regressor"
    shutil.move(str(tabnet_dir), str(archived))

    resolved_payload = json_payload(active)
    atomic_bytes(resolved_path, resolved_payload)
    manifest["contract"]["config_sha256"] = new_config_sha256
    manifest["artifacts"]["resolved_configuration"] = {
        "path": "inputs/params.resolved.json",
        "bytes": len(resolved_payload),
        "sha256": hashlib.sha256(resolved_payload).hexdigest(),
    }
    amendment = {
        "applied_utc": timestamp.isoformat(),
        "scope": "tabnet_regressor_only",
        "basis": "CPU_computational_feasibility_before_retaining_any_TabNet_outer_fold_result",
        "previous_config_sha256": OLD_CONFIG_SHA256,
        "accepted_config_sha256": new_config_sha256,
        "invalidated_artifacts": sorted(EXPECTED_TABNET_FILES),
        "archived_outside_run_root": archived.relative_to(ROOT).as_posix(),
        "previous_tabnet_protocol": old_models["tabnet_regressor"],
        "accepted_tabnet_protocol": new_models["tabnet_regressor"],
        "retained_common_selection_budget": {
            "outer_folds": 5,
            "inner_folds": 4,
            "trials_per_study": 100,
        },
        "retained_mask_functions": ["sparsemax", "entmax"],
        "retained_training_data_policy": "all_rows_in_each_current_training_partition",
        "validation_based_early_stopping": False,
        "device": "cpu",
    }
    manifest.setdefault("protocol_amendments", []).append(amendment)
    manifest["updated_utc"] = timestamp.isoformat()
    atomic_bytes(manifest_path, json_payload(manifest))

    if sha256_json(load_json(resolved_path)) != new_config_sha256:
        raise RuntimeError("Post-migration resolved configuration does not reproduce.")
    if load_json(manifest_path)["contract"]["config_sha256"] != new_config_sha256:
        raise RuntimeError("Post-migration manifest contract does not reproduce.")
    if tabnet_dir.exists():
        raise RuntimeError("The invalidated TabNet directory remains in the run root.")

    print(json.dumps({
        "status": "migrated",
        "run_root": str(RUN_ROOT),
        "new_config_sha256": new_config_sha256,
        "backup_dir": str(backup_dir),
        "invalidated_tabnet_files": sorted(EXPECTED_TABNET_FILES),
    }, indent=2))


if __name__ == "__main__":
    main()
