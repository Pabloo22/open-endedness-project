# Merging Intrinsic and Extrinsic Rewards in Reinforcement Learning

## Repo Structure
```bash
.
├── LICENSE
├── notebooks
│   └── debugging.ipynb  # Used for data exploration and quick prototyping
├── poetry.lock
├── pyproject.toml
├── CONTRIBUTING.md  # Collaboration guidelines for the project
├── README.md
└── src
    └── crew  # Reusable code for the project. It has type-hints and docstrings.
        ├── __init__.py
```

## Installation
We use Poetry version 1.x to manage dependencies and virtual environments. To install the project, run the following command in the terminal:
```
poetry install
```

Jax is not included because it has different installation instructions depending on the platform and hardware. For example, if you have an Nvidia GPU with CUDA 13, run:
```
poetry shell
pip install -U "jax[cuda13]"
```
