# Installation

## Core Installation

Install the package with its core dependencies only:

```bash
pip install customhys
```

This installs the following core dependencies:

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `scipy` | Scientific algorithms (quasi-random sequences, statistics) |
| `matplotlib` | Plotting benchmark functions and results |
| `pandas` | Data manipulation |
| `tqdm` | Progress bars |
| `optproblems` | CEC 2005 benchmark functions |
| `scikit-learn` | Machine-learning utilities (KDE, grid search) |

## Optional Extras

CUSTOMHyS organises optional dependencies into *extras* that you can install
individually or in combination.

### Machine Learning support

```bash
pip install customhys[ml]
```

Adds TensorFlow for neural-network-powered hyper-heuristics (see
{mod}`customhys.machine_learning`). On macOS Apple Silicon the package
automatically installs `tensorflow-macos` and `tensorflow-metal`.

### Development tools

```bash
pip install customhys[dev]
```

Includes `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`, and `pre-commit`.

### Jupyter / Notebook examples

```bash
pip install customhys[examples]
```

Adds `jupyter`, `jupyterlab`, `ipywidgets`, and `notebook`.

### Documentation building

```bash
pip install customhys[docs]
```

Adds `sphinx`, `sphinx-rtd-theme`, and `myst-parser`.

### Everything at once

```bash
pip install customhys[all]
```

## Development Installation (from source)

```bash
# Clone the repository
git clone https://github.com/jcrvz/customhys.git
cd customhys

# Option A – using UV (recommended, 10-100× faster)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev --extra ml --extra examples

# Option B – using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml,examples]"
```

After installation you can verify everything works:

```bash
make validate-setup   # quick health-check
make test             # run the test suite
```

## Apple Silicon (M1 / M2 / M3) Note

TensorFlow requires a special installation path on Apple Silicon Macs.
If the `pip install customhys[ml]` route fails, install TensorFlow via Conda first:

```bash
conda install -c apple tensorflow-deps
```

See [Install TensorFlow on Mac M1/M2 with GPU support](https://medium.com/mlearning-ai/install-tensorflow-on-mac-m1-m2-with-gpu-support-c404c6cfb580)
for more details.

## Python Version Support

CUSTOMHyS supports **Python 3.10, 3.11, and 3.12**.
