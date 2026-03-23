"""Versioned tuning presets and sweep spaces for phased hyperparameter search.

Append new ``*_V2``, ``*_V3``, ... constants instead of editing older ones so
the previous tuning states remain available for reference. Update the
``ACTIVE_*`` aliases when a new version becomes the current default.
"""

from __future__ import annotations

from typing import Any


def get_ngu_base_config_v1() -> dict[str, Any]:
    """Returns the specific hyperparameter values for the NGU module"""
    return {
        "ngu.encoder_mode": "flat_symbolic",
        "ngu.head_activation": "relu",
        "ngu.head_hidden_dim": 64,
        "ngu.episodic_memory_capacity": 4096,
        "ngu.embedding_network_lr": 1e-4,  # Constant learning rate as NGU reward comes from the episodic memory, not a learned predictor
        "ngu.embedding_update_epochs": 1,
        "ngu.embedding_num_minibatches": 64,  # No need, only for performance
        "ngu.gae_lambda": 0.95,
    }


def get_ngu_base_config_v2() -> dict[str, Any]:
    return {
        **get_ngu_base_config_v1(),
        "ngu.kernel_epsilon": 1e-3,
        "ngu.num_neighbors": 10,
    }


def get_ngu_search_space_v1() -> dict[str, Any]:
    """Search space for the NGU phase, comments indicate importance based on heuristic guess."""
    # Total coverage is 486 combos for grid search. 50 for random search could be enough.
    return {
        "ngu.num_neighbors": {"values": [5, 10, 20]},  # High
        "ngu.kernel_epsilon": {"values": [1e-4, 1e-3, 1e-2]},  # High
        "ngu.gamma": {"values": [0.99, 0.995, 0.999]},  # Med
        "ngu.output_embedding_dim": {"values": [32, 64, 128]},  # Med
        "ngu.kernel_cluster_distance": {"values": [1e-4, 1e-3]},  # Low
        "baseline_fixed_training_alpha": {"values": [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3)]},  # High
    }


def get_ngu_search_space_v2() -> dict[str, Any]:
    return {
        "ngu.gamma": {"values": [0.997, 0.998, 0.999]},
        "ngu.output_embedding_dim": {"values": [32, 64]},
        "ngu.kernel_cluster_distance": {"values": [5e-5, 1e-4]},
        "baseline_fixed_training_alpha": {"values": [(0.9, 0.1), (0.85, 0.15), (0.8, 0.2)]},
        "baseline_fixed_training_alpha": {
            "values": [
                (0.999, 0.001),
                (0.995, 0.005),
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
