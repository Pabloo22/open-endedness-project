from pathlib import Path

from crew.experiments.identity import build_experiment_identity


def _resolve_artifacts_root(artifacts_root: str | Path) -> Path:
    return Path(artifacts_root).expanduser().resolve()


def _build_path_from_config(config, artifact_subdir: str) -> Path:
    root = _resolve_artifacts_root(config.artifacts_root)
    identity = build_experiment_identity(config)
    return (
        root
        / artifact_subdir
        / identity.task_identifier
        / identity.algorithm_id
        / identity.intrinsic_rewards_used
        / f"seed{config.train_seed}"
    )


def build_trained_weights_path(config) -> Path:
    return _build_path_from_config(config=config, artifact_subdir="training_results")


def build_best_weights_rollouts_path(config) -> Path:
    return _build_path_from_config(config=config, artifact_subdir="best_weights_rollouts")
