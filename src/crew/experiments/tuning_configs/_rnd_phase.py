"""Versioned tuning presets and sweep spaces for phased hyperparameter search.

Append new ``*_V2``, ``*_V3``, ... constants instead of editing older ones so
the previous tuning states remain available for reference. Update the
``ACTIVE_*`` aliases when a new version becomes the current default.
"""

from __future__ import annotations
from typing import Any

from crew.experiments.tuning_configs import get_default_base_params


DEFAULT_BASELINE_INTRINSIC_ALPHA = 0.2
NUM_ENVIRONMENTS_PER_BATCH = 1024  # Adapted for 16GB GPU memory
DEFAULT_INTRINSIC_MODULES = ("rnd",)

RND_INTRINSIC_BASE_CONFIG_V1: dict[str, Any] = {
    **get_default_base_params(),
    "selected_intrinsic_modules": ("rnd",),
    "baseline_fixed_training_alpha": (
        1.0 - DEFAULT_BASELINE_INTRINSIC_ALPHA,
        DEFAULT_BASELINE_INTRINSIC_ALPHA,
    ),
}


def get_rnd_base_config_v1():
    return {
        "selected_intrinsic_modules": ("rnd",),
        "baseline_fixed_training_alpha": (
            1.0 - DEFAULT_BASELINE_INTRINSIC_ALPHA,
            DEFAULT_BASELINE_INTRINSIC_ALPHA,
        ),
    }


ACTIVE_INTRINSIC_BASE_CONFIGS: dict[str, dict[str, Any]] = {
    "rnd": RND_INTRINSIC_BASE_CONFIG_V1,
}


def get_rnd_search_space_v1() -> dict[str, Any]:
    return {
        "rnd.predictor_network_lr": {"values": [5e-5, 1e-4, 2e-4]},
        "rnd.output_embedding_dim": {"values": [128, 256]},
        "rnd.head_hidden_dim": {"values": [128, 256]},
        "rnd.gamma": {"values": [0.99, 0.995, 0.999]},
        "rnd.gae_lambda": {"values": [0.95, 0.99]},
        "baseline_fixed_training_alpha": {"values": [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3)]},
    }
