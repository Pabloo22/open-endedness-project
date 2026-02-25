from pathlib import Path


def _resolve_artifacts_root(artifacts_root: str | Path) -> Path:
    return Path(artifacts_root).expanduser().resolve()


def build_trained_weights_path(
    algorithm_id, env_id, seed=None, artifacts_root="artifacts"
):
    root = _resolve_artifacts_root(artifacts_root)
    path = root / f"training_results/{algorithm_id}/{env_id}/seed{seed}"
    return path


def build_best_weights_rollouts_path(
    algorithm_id, env_id, seed=None, artifacts_root="artifacts"
):
    root = _resolve_artifacts_root(artifacts_root)
    path = root / f"best_weights_rollouts/{algorithm_id}/{env_id}/seed{seed}"
    return path
