"""Versioned tuning presets and sweep spaces for phased hyperparameter search.

Append new ``*_V2``, ``*_V3``, ... constants instead of editing older ones so
the previous tuning states remain available for reference. Update the
``ACTIVE_*`` aliases when a new version becomes the current default.
"""

from __future__ import annotations

from typing import Any


def get_rnd_base_config_v1() -> dict[str, Any]:
    """Returns the specific hyperparameter values for the RND module"""
    return {
        "rnd.output_embedding_dim": 256,
        "rnd.head_activation": "relu",
        "rnd.head_hidden_dim": 256,
        "rnd.predictor_network_lr": 1e-4,
        "rnd.predictor_update_epochs": 1,
        "rnd.predictor_num_minibatches": 64,
        "rnd.gamma": 0.99,
        "rnd.gae_lambda": 0.95,
    }


def get_rnd_base_config_v2() -> dict[str, Any]:
    """Returns the specific hyperparameter values for the RND module"""
    return {
        **get_rnd_base_config_v1(),
        "rnd.gae_lambda": 0.99,
        "rnd.gamma": 0.995,
        "rnd.predictor_num_minibatches": 32,
    }


def get_rnd_search_space_v1() -> dict[str, Any]:
    return {
        "rnd.predictor_network_lr": {"values": [1e-5, 5e-5, 8e-5, 1e-4, 3e-4, 5e-4]},
        "rnd.predictor_update_epochs": {"values": [1, 3, 5]},
        "rnd.predictor_num_minibatches": {"values": [32, 64, 128]},
        "rnd.gamma": {"values": [0.99, 0.995, 0.999]},
        "rnd.gae_lambda": {"values": [0.95, 0.99]},
    }


def get_rnd_search_space_v2() -> dict[str, Any]:
    return {
        "rnd.predictor_network_lr": {"values": [1e-5, 5e-5, 1e-4, 5e-4]},
        "rnd.predictor_update_epochs": {"values": [1, 3]},
        "rnd.predictor_num_minibatches": {"values": [32, 64]},
        "rnd.gamma": {"values": [0.99, 0.995]},
        "baseline_fixed_training_alpha": {
            "values": [
                (0.99, 0.01),
                (0.95, 0.05),
                (0.9, 0.1),
                (0.8, 0.2),
                (0.7, 0.3),
                (0.6, 0.4),
                (0.5, 0.5),
                (0.4, 0.6),
                (0.3, 0.7),
                (0.2, 0.8),
                (0.1, 0.9),
                (0.0, 1.0),
            ]
        },
    }
