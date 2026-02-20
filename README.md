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
or
```
poetry install --with notebooks
```
if you want to include the dependencies for the notebooks.

Jax is not included because it has different installation instructions depending on the platform and hardware. For example, if you have an Nvidia GPU with CUDA 13, run:
```
poetry shell
pip install -U "jax[cuda13]"
```

 ## Acknowledgments

 The code for the policies' base Transformer-XL architecture, and the code in `ppo_update.py` is inspired from [transformerXL_PPO_JAX](https://github.com/Reytuag/transformerXL_PPO_JAX/tree/main), which itself acknowledges the following sources of inspiration:
- [PureJaxRL](https://github.com/luchris429/purejaxrl)
- [Huggingface transformerXL](https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/deprecated/transfo_xl/modeling_transfo_xl.py)
- https://github.com/MarcoMeter/episodic-transformer-memory-ppo
