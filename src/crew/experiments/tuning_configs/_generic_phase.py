"""Versioned tuning presets and sweep spaces for phased hyperparameter search.

Append new ``*_V2``, ``*_V3``, ... constants instead of editing older ones so
the previous tuning states remain available for reference. Update the
``ACTIVE_*`` aliases when a new version becomes the current default.
"""

from __future__ import annotations

from typing import Any

from crew.experiments.constants import CRAFTAX_CLASSIC_INTERMEDIATE_ACHIEVEMENT_IDS


DEFAULT_BASELINE_INTRINSIC_ALPHA = 0.2
NUM_ENVIRONMENTS_PER_BATCH = 1024  # Adapted for 16GB GPU memory


# Default params
def get_default_base_params() -> dict[str, Any]:
    return {
        "env_id": "Craftax-Classic-Symbolic-v1",
        "achievement_ids_to_block": CRAFTAX_CLASSIC_INTERMEDIATE_ACHIEVEMENT_IDS,
        "training_mode": "baseline",
        "episode_max_steps": 4096,
        "remove_health_reward": True,
        "selected_intrinsic_modules": (),
        "baseline_fixed_training_alpha": (1.0,),
        "num_envs_per_batch": NUM_ENVIRONMENTS_PER_BATCH,
        "num_steps_per_env": 8192,
        "num_steps_per_update": 256,
        "eval_every_n_batches": 1,
        "eval_num_envs": 64,
        "eval_num_episodes": 2,
        "evaluation_alphas": ((1.0,),),
        "update_epochs": 1,
        "num_minibatches": 16,
        "obs_emb_dim": 256,
        "past_context_length": 128,
        "subsequence_length_in_loss_calculation": 64,
        "num_attn_heads": 4,
        "num_transformer_blocks": 2,
        "transformer_hidden_states_dim": 192,
        "qkv_features": 192,
        "head_hidden_dim": 256,
        "enable_wandb": False,
        "is_timing_run": False,
    }


# FAST SETTING: scaled down the network and the length of the history it looks at
def get_lightweight_base_params() -> dict[str, Any]:
    return {
        **get_default_base_params(),
        "obs_emb_dim": 128,
        "past_context_length": 64,  # 128 -> 64
        "subsequence_length_in_loss_calculation": 32,  # 64 -> 32
        "num_transformer_blocks": 1,  # 2 -> 1
        "transformer_hidden_states_dim": 128,  # 192 -> 128
        "head_hidden_dim": 128,  # 256 -> 128
    }


def get_base_params_after_generic_sweep():
    return {
        **get_default_base_params(),
        "clip_eps": 0.2,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
    }


def get_best_generic_params() -> dict[str, Any]:
    return {
        **get_default_base_params(),
        "clip_eps": 0.2,
        "gae_lambda": 0.95,
        "ent_coef": 0.005,
        "lr": 0.0005,
    }


def get_generic_search_space_v1() -> dict[str, Any]:
    return {
        "lr": {"values": [5e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4]},
        "ent_coef": {"values": [0.0, 0.005, 0.01, 0.015, 0.02]},
        "clip_eps": {"values": [0.1, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3]},
        "gae_lambda": {"values": [0.95, 0.975, 0.99]},
    }


def get_generic_search_space_v2() -> dict[str, Any]:
    return {
        "lr": {"values": [4e-4, 0.0005, 0.0007, 0.0009]},
        "ent_coef": {"values": [0.005, 0.01]},
        "train_seed": {"values": [1, 2, 3, 4, 5]},
    }
