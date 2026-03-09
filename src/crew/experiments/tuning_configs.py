"""Versioned tuning presets and sweep spaces for phased hyperparameter search.

Append new ``*_V2``, ``*_V3``, ... constants instead of editing older ones so
the previous tuning states remain available for reference. Update the
``ACTIVE_*`` aliases when a new version becomes the current default.
"""

from __future__ import annotations

import copy
from typing import Any

from crew.experiments.constants import CRAFTAX_CLASSIC_INTERMEDIATE_ACHIEVEMENT_IDS


def _divide_or_raise_error(numerator: int, denominator: int) -> int:
    """Helper for validating that num_envs_per_batch is divisible by num_minibatches."""
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}.")
    return numerator // denominator


DEFAULT_BASELINE_INTRINSIC_ALPHA = 0.2
NUM_ENVIRONMENTS_PER_BATCH = 64
DEFAULT_INTRINSIC_MODULES = ("rnd",)

GENERIC_BASE_CONFIG_V1: dict[str, Any] = {
    "env_id": "Craftax-Classic-Symbolic-v1",
    "achievement_ids_to_block": CRAFTAX_CLASSIC_INTERMEDIATE_ACHIEVEMENT_IDS,
    "training_mode": "baseline",
    "remove_health_reward": True,
    "selected_intrinsic_modules": (),
    "baseline_fixed_training_alpha": (1.0,),
    "num_envs_per_batch": 64,
    "num_steps_per_env": 512,
    "num_steps_per_update": 256,
    "eval_every_n_batches": 1,
    "eval_num_envs": 64,
    "eval_num_episodes": 2,
    "evaluation_alphas": ((1.0,),),
    "update_epochs": 1,
    "num_minibatches": 16,
    "obs_emb_dim": 256,
    "past_context_length": 64,
    "subsequence_length_in_loss_calculation": 64,
    "num_attn_heads": 4,
    "num_transformer_blocks": 1,
    "transformer_hidden_states_dim": 64,
    "qkv_features": 64,
    "head_hidden_dim": 64,
    "enable_wandb": False,
    "is_timing_run": False,
}

RND_INTRINSIC_BASE_CONFIG_V1: dict[str, Any] = {
    **GENERIC_BASE_CONFIG_V1,
    "selected_intrinsic_modules": ("rnd",),
    "baseline_fixed_training_alpha": (
        1.0 - DEFAULT_BASELINE_INTRINSIC_ALPHA,
        DEFAULT_BASELINE_INTRINSIC_ALPHA,
    ),
}

RND_CURRICULUM_BASE_CONFIG_V1: dict[str, Any] = {
    **GENERIC_BASE_CONFIG_V1,
    "training_mode": "curriculum",
    "selected_intrinsic_modules": ("rnd",),
    "baseline_fixed_training_alpha": None,
    "evaluation_alphas": (
        (1.0 - DEFAULT_BASELINE_INTRINSIC_ALPHA, DEFAULT_BASELINE_INTRINSIC_ALPHA),
        (1.0, 0.0),
    ),
}

ACTIVE_GENERIC_BASE_CONFIG = GENERIC_BASE_CONFIG_V1
ACTIVE_INTRINSIC_BASE_CONFIGS: dict[str, dict[str, Any]] = {
    "rnd": RND_INTRINSIC_BASE_CONFIG_V1,
}
ACTIVE_CURRICULUM_BASE_CONFIGS: dict[tuple[str, ...], dict[str, Any]] = {
    ("rnd",): RND_CURRICULUM_BASE_CONFIG_V1,
}

GENERIC_SEARCH_SPACE_V1: dict[str, Any] = {
    "lr": {"values": [5e-3, 1e-4, 2e-4, 3e-4, 5e-4]},
    "ent_coef": {"values": [0.0, 0.005, 0.01, 0.02]},
    "clip_eps": {"values": [0.1, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3]},
    "gamma": {"values": [0.99, 0.995, 0.999]},
    "gae_lambda": {"values": [0.95, 0.99]},
    "update_epochs": {"values": [1, 2, 4]},
    "num_minibatches": {
        "values": [
            _divide_or_raise_error(NUM_ENVIRONMENTS_PER_BATCH, 8),
            _divide_or_raise_error(NUM_ENVIRONMENTS_PER_BATCH, 4),
            _divide_or_raise_error(NUM_ENVIRONMENTS_PER_BATCH, 2),
        ]
    },
    "obs_emb_dim": {"values": [128, 256]},
    "past_context_length": {"values": [64, 128]},
    "num_transformer_blocks": {"values": [1, 2]},
    "transformer_hidden_states_dim": {"values": [64, 128, 192]},
    "head_hidden_dim": {"values": [64, 128, 256]},
    "inject_alpha_at_trunk": {"values": [True, False]},
}

RND_INTRINSIC_SEARCH_SPACE_V1: dict[str, Any] = {
    "rnd.predictor_network_lr": {"values": [5e-5, 1e-4, 2e-4]},
    "rnd.output_embedding_dim": {"values": [128, 256]},
    "rnd.head_hidden_dim": {"values": [128, 256]},
    "rnd.gamma": {"values": [0.99, 0.995, 0.999]},
    "rnd.gae_lambda": {"values": [0.95, 0.99]},
    "baseline_fixed_training_alpha": {"values": [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3)]},
}

CURRICULUM_SEARCH_SPACE_V1: dict[str, Any] = {
    "curriculum.score_lambda": {"values": [0.0, 0.25, 0.5, 0.75, 1.0]},
    "curriculum.predictor_lr": {"values": [5e-5, 1e-4, 2e-4]},
    "curriculum.lp_norm_ema_beta": {"values": [0.02, 0.05, 0.1]},
}

ACTIVE_GENERIC_SEARCH_SPACE = GENERIC_SEARCH_SPACE_V1
ACTIVE_INTRINSIC_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "rnd": RND_INTRINSIC_SEARCH_SPACE_V1,
}
ACTIVE_CURRICULUM_SEARCH_SPACE = CURRICULUM_SEARCH_SPACE_V1


def get_generic_base_config() -> dict[str, Any]:
    """Return the current generic-phase base config."""
    return copy.deepcopy(ACTIVE_GENERIC_BASE_CONFIG)


def get_intrinsic_base_config(intrinsic_module: str) -> dict[str, Any]:
    """Return the current intrinsic-phase base config for one module."""
    if intrinsic_module not in ACTIVE_INTRINSIC_BASE_CONFIGS:
        msg = (
            f"Unsupported intrinsic module {intrinsic_module!r} for intrinsic tuning presets. "
            f"Supported modules: {tuple(ACTIVE_INTRINSIC_BASE_CONFIGS)}."
        )
        raise ValueError(msg)
    return copy.deepcopy(ACTIVE_INTRINSIC_BASE_CONFIGS[intrinsic_module])


def get_curriculum_base_config_for_modules(module_names: tuple[str, ...]) -> dict[str, Any]:
    """Return the current curriculum-phase base config for one intrinsic-module set."""
    normalized_module_names = normalize_intrinsic_modules(module_names)
    if normalized_module_names not in ACTIVE_CURRICULUM_BASE_CONFIGS:
        msg = (
            f"Unsupported intrinsic modules {normalized_module_names!r} for curriculum tuning presets. "
            f"Supported modules: {tuple(ACTIVE_CURRICULUM_BASE_CONFIGS)}."
        )
        raise ValueError(msg)
    return copy.deepcopy(ACTIVE_CURRICULUM_BASE_CONFIGS[normalized_module_names])


def get_generic_search_space() -> dict[str, Any]:
    """Return the current generic-phase search space."""
    return copy.deepcopy(ACTIVE_GENERIC_SEARCH_SPACE)


def get_intrinsic_search_space(intrinsic_module: str) -> dict[str, Any]:
    """Return the current intrinsic-phase search space for one module."""
    if intrinsic_module not in ACTIVE_INTRINSIC_SEARCH_SPACES:
        msg = (
            f"Unsupported intrinsic module {intrinsic_module!r} for intrinsic search spaces. "
            f"Supported modules: {tuple(ACTIVE_INTRINSIC_SEARCH_SPACES)}."
        )
        raise ValueError(msg)
    return copy.deepcopy(ACTIVE_INTRINSIC_SEARCH_SPACES[intrinsic_module])


def get_curriculum_search_space() -> dict[str, Any]:
    """Return the current curriculum-phase search space."""
    return copy.deepcopy(ACTIVE_CURRICULUM_SEARCH_SPACE)


def normalize_intrinsic_modules(module_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return a stable intrinsic-module tuple for preset lookups."""
    if not module_names:
        msg = "At least one intrinsic module must be provided."
        raise ValueError(msg)
    return tuple(sorted(module_names))
