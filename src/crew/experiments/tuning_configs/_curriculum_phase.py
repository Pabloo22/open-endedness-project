"""Versioned tuning presets and sweep spaces for phased hyperparameter search.

Append new ``*_V2``, ``*_V3``, ... constants instead of editing older ones so
the previous tuning states remain available for reference. Update the
``ACTIVE_*`` aliases when a new version becomes the current default.
"""

from __future__ import annotations

from typing import Any


DEFAULT_INTRINSIC_MODULES = ("rnd",)


def get_curriculum_search_space_v1() -> dict[str, Any]:
    return {
        "curriculum.score_lambda": {"values": [0.0, 0.25, 0.5, 0.75, 1.0]},
        "curriculum.predictor_lr": {"values": [5e-5, 1e-4, 2e-4]},
        "curriculum.lp_norm_ema_beta": {"values": [0.02, 0.05, 0.1]},
    }
