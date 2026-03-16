"""Versioned tuning presets and sweep spaces for phased hyperparameter search.

Append new ``*_V2``, ``*_V3``, ... constants instead of editing older ones so
the previous tuning states remain available for reference. Update the
``ACTIVE_*`` aliases when a new version becomes the current default.
"""

from __future__ import annotations

from typing import Any

from crew.experiments.tuning_configs import default_base_params


DEFAULT_BASELINE_INTRINSIC_ALPHA = 0.2
DEFAULT_INTRINSIC_MODULES = ("ngu",)

NGU_INTRINSIC_BASE_CONFIG_V1: dict[str, Any] = {
    **default_base_params(),
    "selected_intrinsic_modules": ("ngu",),
    "baseline_fixed_training_alpha": (
        1.0 - DEFAULT_BASELINE_INTRINSIC_ALPHA,
        DEFAULT_BASELINE_INTRINSIC_ALPHA,
    ),
}


def get_ngu_base_config_v1() -> dict[str, Any]:
    return {
        "selected_intrinsic_modules": ("ngu",),
        "baseline_fixed_training_alpha": (
            1.0 - DEFAULT_BASELINE_INTRINSIC_ALPHA,
            DEFAULT_BASELINE_INTRINSIC_ALPHA,
        ),
    }


def get_ngu_search_space_v1() -> dict[str, Any]:
    return {
        "baseline_fixed_training_alpha": {"values": [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3)]},
    }
