# ICSOR Article Benchmark

The article benchmark is a single, restartable workflow in [main.ipynb](main.ipynb). Configuration lives in [config/params.json](config/params.json); [config/paths.json](config/paths.json) contains only repository-relative input and result paths.

The study evaluates nine conventional surrogate methods: XGBoost, LightGBM, CatBoost, AdaBoost, Random Forest, support vector regression, k-nearest neighbors, partial least squares, and a multilayer perceptron. The workflow runs entirely on multicore CPUs and uses no external pretrained weights.

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- A multicore CPU and sufficient storage for the full result bundle

Install the tracked environment from the repository root:

```powershell
uv sync --frozen
```

## Run the notebook

The default `full` profile uses run ID `article_final_v1`. It generates 10,000 accepted in-distribution states and 600 accepted states in each of the mild and severe extrapolation regimes, then performs the complete nine-model nested benchmark.

For a short integration check:

```powershell
$env:ICSOR_PROFILE = "smoke"
$env:ICSOR_RUN_ID = "article_smoke_local"
New-Item -ItemType Directory -Force results/executed_notebooks | Out-Null
uv run jupyter nbconvert --execute --to notebook main.ipynb `
  --output main.smoke.executed.ipynb `
  --output-dir results/executed_notebooks `
  --ExecutePreprocessor.timeout=-1
```

For the authoritative analysis:

```powershell
$env:ICSOR_PROFILE = "full"
Remove-Item Env:ICSOR_RUN_ID -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force results/executed_notebooks | Out-Null
uv run jupyter nbconvert --execute --to notebook main.ipynb `
  --output main.full.executed.ipynb `
  --output-dir results/executed_notebooks `
  --ExecutePreprocessor.timeout=-1
```

An interactive execution should use a copy outside the immutable run root so that outputs do not change the source-notebook hash:

```powershell
$env:ICSOR_PROFILE = "full"
New-Item -ItemType Directory -Force results/interactive | Out-Null
Copy-Item main.ipynb results/interactive/main.session.ipynb
uv run jupyter lab --ServerApp.root_dir=. results/interactive/main.session.ipynb
```

The full run is computationally substantial. It performs 100 sequential TPE trials for every outer-fold model selection and a separate 100-trial full-data search for each of nine methods, in addition to mechanistic sampling, evaluation at eleven nested sizes, timing, and OOD refits. Allow multi-day wall time; matching-contract resume is designed for interruptions.

## Results and resume behavior

Each run writes beneath `results/{run_id}`. Resume is permitted only when the configuration, input workbook, paths, source notebook, dependency lock, and recorded environment contract match. Completed runs are immutable.

Smoke outputs always carry `article_eligible=false`. Only a successfully completed full profile can supply article values.

The run root contains input snapshots, accepted datasets and every solver attempt, matrix operators, split assignments, tuning databases and trial tables, row-level raw and projected predictions, component and composite metrics, timing repetitions, fitted model bundles, and publication assets. `manifest.json` inventories and hashes the completed bundle; `manifest.sha256` checks the manifest itself.

The scientific writing workspace is `artifacts/wip/`. Publication numbers and figures must be copied only from the completed full result bundle and verified against its asset inventory before the manuscript and supplement are rebuilt.
