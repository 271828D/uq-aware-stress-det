# Uncertainty Quantification Aware Stress Detection

Binary stress classification using ML baselines with **subject-aware splits**.

## Quick Start

### 1. Install `uv` (Fast Python Package Manager)
`uv` is an extremely fast Python package installer and resolver written in Rust.

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Verify installation:
```bash
uv --version
```

### 2. Clone & Setup Environment
Navigate to your project folder and create a virtual environment:

```Bash
# Create a venv using uv
uv venv

# Activate the venv
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
Install core depenndencies from `pyproject.toml`:
```Bash
uv pip install -e .
```

### 4. Configure Data Path
Open `conf/data/stress_data.yaml` and ensure the `source_file` path points to your actual CSV location:
```
data:
  source_file: "/absolute/path/to/your/data.csv"
```
### 5. Run Training (Single Run)
Run the trainer with a specific model (e.g., Logistic Regression):
```bash
uv run python src/train/trainer.py model=logistic_regression seed=69
```
### 6. Multi-Run Training with Hyperparameter Opt.
Run the trainer with a specific model (e.g., Logistic Regression) and do hyper-parameter search. `Only available with logistic_regression and random_forest`:
```bash
uv run python src/train/trainer.py model=logistic_regression search_space=logistic_regression
```
