"""Versioned tuning presets and sweep spaces for phased hyperparameter search.

Append new ``*_V2``, ``*_V3``, ... constants instead of editing older ones so
the previous tuning states remain available for reference. Update the
``ACTIVE_*`` aliases when a new version becomes the current default.
"""

from __future__ import annotations

from typing import Any


def get_ngu_base_config_v1() -> dict[str, Any]:
    """Returns the specific hyperparameter values for the NGU module"""
    return {}


def get_ngu_search_space_v1() -> dict[str, Any]:
    return {}
