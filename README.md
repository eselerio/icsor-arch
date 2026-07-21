# ICSOR Article Benchmark

The article benchmark is a single, restartable workflow in [main.ipynb](main.ipynb). Configuration lives in [config/params.json](config/params.json); [config/paths.json](config/paths.json) contains only repository-relative input and result paths.

The study evaluates 13 CPU-based surrogate methods: XGBoost, LightGBM, CatBoost, AdaBoost, Random Forest, Extra Trees, support vector regression, k-nearest neighbors, partial least squares (PLS), Multi-task Elastic Net, Multi-task Lasso, a multilayer perceptron, and TabNet. PLS and the two multi-task linear estimators form the prespecified `Interpretable models` reporting category. Every method predicts the same 20-component effluent vector; the workflow uses no GPU or external pretrained weights.

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- A multicore CPU and sufficient storage for the full result bundle

Install the tracked environment from the repository root:

```powershell
uv sync --frozen
```

## Run the notebook

The default `full` profile uses run ID `article_final_v2`. It generates 10,000 accepted in-distribution states and 600 accepted states in each of the mild and severe extrapolation regimes, then performs the complete 13-model nested benchmark.

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

The full run is computationally substantial. It performs 100 sequential TPE trials for every outer-fold model selection and a separate 100-trial full-data search for each of 13 methods, in addition to mechanistic sampling, evaluation at eleven nested sizes, timing, and OOD refits. TabNet uses deterministic CPU PyTorch operations, 20--50 selected epochs, widths 8--24, three or four attentive steps, and batch sizes 512 or 1024; early stopping is disabled. SVR is CPU-bounded to radial and linear kernels, shrinking enabled, and 20,000 solver iterations per component regressor. MLP uses every current training row, 250 maximum epochs, and 15-epoch patience based on training-loss improvement; validation-based early stopping is disabled. The accepted ranges are recorded in `config/params.json`. Allow multi-day wall time. Matching-contract resume safely reuses completed units, but do not deliberately terminate an active Optuna trial: an unclean interruption can leave a stale `RUNNING` database row that must be reconciled before strict completion validation.

## Results and resume behavior

Each run writes beneath `results/{run_id}`. Resume is permitted only when the configuration, input workbook, paths, source notebook, dependency lock, and recorded environment contract match. Completed runs are immutable.

Smoke outputs always carry `article_eligible=false`. Only a successfully completed full profile can supply article values.

The run root contains input snapshots, accepted datasets and every solver attempt, matrix operators, split assignments, tuning databases and trial tables, row-level raw and projected predictions, component and composite metrics, timing repetitions, fitted model bundles, and publication assets. The fitted bundles are internal run artifacts whose notebook-defined wrapper classes are reloaded by the same execution kernel for OOD evaluation; they are not advertised as standalone deployment packages. For trusted artifacts, `load_surrogate_bundle()` in `scripts/load_surrogate_bundle.py` provides a bounded inference-compatibility loader and a read-only check that loads all 13 bundles and compares fresh predictions with the stored OOD rows:

```powershell
uv run python scripts/load_surrogate_bundle.py `
  --smoke-check-run-root results/article_smoke_13model_final_v2
```

The helper optionally verifies a manifest-recorded SHA-256 digest, but joblib remains pickle-based and must not be used with untrusted files. `manifest.json` inventories and hashes the completed bundle; `manifest.sha256` checks the manifest itself.

The scientific writing workspace is `artifacts/wip/`. Publication numbers and figures must be copied only from the completed full result bundle and verified against its asset inventory before the manuscript and supplement are rebuilt.

After the full run has written `COMPLETED.json`, build the manuscript-ready audit bundle with:

```powershell
uv run python scripts/build_final_paper_assets.py `
  --run-root results/article_final_v2 `
  --output-dir artifacts/wip/generated/article_final_v2
```

The finalizer is read-only with respect to the immutable run. It verifies every manifest hash, independently reproduces metrics and physical diagnostics from prediction rows, checks all 78 searches and expected table grids, and writes source CSVs, LaTeX row fragments, vector figures, and a derived-asset hash manifest outside `results/article_final_v2`.
