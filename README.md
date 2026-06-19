# ICSOR Workspace

This project uses `uv` to manage the Python environment and dependencies.

## Requirements

- Python 3.11 or 3.12
- `uv` package manager

## 1. Install `uv`

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternative installation methods are available in the official `uv` documentation if you already use `winget`, `pipx`, or another package manager.

## 2. Sync the environment

From the project root, install all dependencies with:

```powershell
uv sync
```

This will create the local virtual environment and install the packages declared in [pyproject.toml](pyproject.toml).

## 3. Open the notebooks

Start Jupyter through `uv` so the project environment is used:

```powershell
uv run jupyter lab
```

If you prefer VS Code notebooks, open the project folder and select the Python kernel from the `.venv` created by `uv sync`.

## 4. Run the notebooks

Run the notebooks in this order:

1. [simulation.ipynb](simulation.ipynb)
2. [comparison.ipynb](comparison.ipynb)

The intended workflow is:

1. Execute all cells in [simulation.ipynb](simulation.ipynb) to generate the simulation outputs.
2. Execute all cells in [comparison.ipynb](comparison.ipynb) to compare results and analyze the generated data.

## Quick Start

```powershell
uv sync
uv run jupyter lab
```

Then run [simulation.ipynb](simulation.ipynb) first, followed by [comparison.ipynb](comparison.ipynb).