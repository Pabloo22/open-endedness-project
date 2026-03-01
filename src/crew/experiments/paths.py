from pathlib import Path


def _resolve_artifacts_root(artifacts_root: str | Path) -> Path:
    return Path(artifacts_root).expanduser().resolve()


def _format_intrinsic_rewards_id(intrinsic_rewards: str | tuple[str, ...] | list[str] | None) -> str:
    if intrinsic_rewards is None:
        return "none"
    if isinstance(intrinsic_rewards, str):
        normalized = intrinsic_rewards.strip()
        return normalized if normalized else "none"

    cleaned_names = tuple(name for name in intrinsic_rewards if name)
    return "+".join(cleaned_names) if cleaned_names else "none"


def _resolve_algorithm_id(training_mode: str) -> str:
    if training_mode in ("curriculum", "baseline"):
        return training_mode

    msg = f"Unsupported training_mode {training_mode!r}. Expected one of ('curriculum', 'baseline')."
    raise ValueError(msg)


def _build_path_from_config(config, artifact_subdir: str) -> Path:
    root = _resolve_artifacts_root(config.artifacts_root)
    algorithm_id = _resolve_algorithm_id(config.training_mode)
    intrinsic_rewards_id = _format_intrinsic_rewards_id(config.selected_intrinsic_modules)
    return root / f"{config.env_id}/{artifact_subdir}/{algorithm_id}/{intrinsic_rewards_id}/seed{config.train_seed}"


def build_trained_weights_path(config) -> Path:
    return _build_path_from_config(config=config, artifact_subdir="training_results")


def build_best_weights_rollouts_path(config) -> Path:
    return _build_path_from_config(config=config, artifact_subdir="best_weights_rollouts")
