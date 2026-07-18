# ICSOR Article Benchmark

The article benchmark is a single, restartable workflow in [main.ipynb](main.ipynb). Configuration lives in [config/params.json](config/params.json), while [config/paths.json](config/paths.json) contains only repository-relative input and result paths.

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- An NVIDIA CUDA GPU with at least 16 GiB VRAM for the complete run, including TabICLv2

Install the locked environment from the repository root:

```powershell
uv sync --frozen
```

## Stage the TabICLv2 checkpoint

The full profile never downloads model weights implicitly. Stage the pinned checkpoint before running it:

```powershell
New-Item -ItemType Directory -Force data/tabicl | Out-Null
uv run hf download jingang/TabICL tabicl-regressor-v2-20260212.ckpt --local-dir data/tabicl
```

The expected file is `data/tabicl/tabicl-regressor-v2-20260212.ckpt`. Verify it before a full run:

```powershell
(Get-Item data/tabicl/tabicl-regressor-v2-20260212.ckpt).Length
(Get-FileHash data/tabicl/tabicl-regressor-v2-20260212.ckpt -Algorithm SHA256).Hash.ToLowerInvariant()
```

The expected size is `114324594` bytes and the expected SHA-256 is `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`.

## Run the notebook

The default profile is `full`, with run ID `article_final_v1`. It generates 10,000 in-distribution samples plus 600 mild and 600 severe out-of-distribution samples, then runs the complete nested benchmark.

For an interactive full run, work on an execution copy so notebook outputs cannot modify the authoritative source while its hash is part of the run contract:

```powershell
$env:ICSOR_PROFILE = "full"
New-Item -ItemType Directory -Force results/interactive | Out-Null
Copy-Item main.ipynb results/interactive/main.session.ipynb
uv run jupyter lab --ServerApp.root_dir=. results/interactive/main.session.ipynb
```

For a quick integration check, select the reduced `smoke` overrides:

```powershell
$env:ICSOR_PROFILE = "smoke"
New-Item -ItemType Directory -Force results/interactive | Out-Null
Copy-Item main.ipynb results/interactive/main.smoke.session.ipynb
uv run jupyter lab --ServerApp.root_dir=. results/interactive/main.smoke.session.ipynb
```

Run the same workflow headlessly with `nbconvert`:

```powershell
$env:ICSOR_PROFILE = "smoke"
New-Item -ItemType Directory -Force results/executed_notebooks | Out-Null
uv run jupyter nbconvert --execute --to notebook main.ipynb --output main.smoke.executed.ipynb --output-dir results/executed_notebooks --ExecutePreprocessor.timeout=-1
```

For the authoritative full run, use the same command with `ICSOR_PROFILE=full` and an output name such as `main.full.executed.ipynb`. Executed notebook copies remain outside the immutable run root.

The full run is intentionally substantial: mechanistic acceptance sampling, 9 conventional methods with 100-trial nested searches, and 20 independent TabICLv2 contexts dominate elapsed time. Allow multi-day wall time on the authoritative workstation; actual duration is hardware-dependent. Run studies sequentially when collecting timings.

## Results and resume behavior

Each run writes beneath `results/{run_id}`. The configuration enables matching-contract resume by default. Configuration, paths, workbook, notebook, dependency lock, checkpoint, and hardware hashes must all agree before an incomplete run can continue. Every completed run is immutable; choose a new run ID for another execution.

The smoke profile is currently a CPU-capable check of simulation, projection, and the conventional-model pipeline only. It disables TabICLv2 because the article configuration fixes TabICLv2 inference to CUDA. A successful CPU smoke run therefore does not validate the foundation-model comparison or constitute a complete article run.

The run root contains immutable input snapshots, accepted datasets and every solver attempt, matrix operators, persistent split assignments, tuning databases and trial tables, row-level raw/projected predictions, metrics, timing repetitions, final model bundles, and publication assets. `manifest.json` inventories and hashes the completed bundle; `manifest.sha256` provides an adjacent integrity checksum for that manifest.

The authoritative writing workspace is `artifacts/wip/`. After the full run completes, verify every CSV, TeX, PDF, and PNG source against `results/article_final_v1/article/asset_inventory.csv`, copy the accepted publication assets into that workspace, and replace every explicitly illustrative value and panel in both documents. Then build both PDFs from a terminal:

```powershell
Push-Location artifacts/wip
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex
Pop-Location
```

Do not remove the draft-results notices until the complete CUDA run is eligible, every displayed value resolves to its source-data row, and both PDFs have been visually inspected.
