"""Run hyperparameter tuning for the training stack with W&B random search.

This module wraps the existing training entrypoint in a Weights & Biases sweep
agent so that W&B can randomly sample hyperparameters and launch repeated
training trials.

Typical workflow:
1. Adjust the search space in ``build_default_sweep_config`` to the config
    fields and value ranges you want to tune.
2. Start a new sweep and agent with a command such as
    ``poetry run python -m crew.experiments.wandb_random_search --count 100 --training-mode baseline``.
3. If you only want to create the sweep and inspect it in W&B before running
    trials, use ``--create-only``.
4. If you already have a sweep id, reuse it with ``--sweep-id <id>``.

Each trial starts from the base configuration built by
``build_base_tuning_config`` and then applies the sampled W&B overrides before
rebuilding ``TrainConfig`` so validation still runs. Trials are executed
through ``run_main_algo_training``.

The script deliberately disables the inner training loop's W&B logging to avoid
nested runs. Instead, each sweep trial owns a single W&B run and logs a compact
final summary, including the main sweep objective
``tuning/objective_eval_return_mean`` and several final training and evaluation
diagnostics. Use ``--save-results`` if you also want each trial to persist the
full checkpoint and training artifacts.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import wandb

from crew.experiments.run_training import run_main_algo_training
from crew.experiments.constants import CRAFTAX_CLASSIC_INTERMEDIATE_ACHIEVEMENT_IDS
from crew.main_algo.config import TrainConfig
from crew.main_algo.logging import build_wandb_run_name


DEFAULT_OBJECTIVE_METRIC = "tuning/objective_eval_return_mean"
NUM_ENVIRONMENTS_PER_BATCH = 64


def main() -> None:
    """Create or reuse a W&B sweep and launch the requested number of trials."""
    args = parse_args()
    base_config = build_base_tuning_config(
        training_mode=args.training_mode,
        project=args.project,
        entity=args.entity,
        group=args.group,
        train_seed=args.train_seed,
        total_timesteps=args.total_timesteps,
    )
    sweep_config = build_default_sweep_config(training_mode=args.training_mode)

    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
        print(f"Created W&B sweep: {sweep_id}")
        if args.create_only:
            return
    else:
        print(f"Using existing W&B sweep: {sweep_id}")

    wandb.agent(
        sweep_id,
        function=lambda: _run_single_trial(base_config=base_config, save_results=args.save_results),
        count=args.count,
        project=args.project,
        entity=args.entity,
    )


def build_default_sweep_config(
    *,
    training_mode: str,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
) -> dict[str, Any]:
    """Return the default W&B random-search sweep definition for the selected mode."""
    parameters: dict[str, Any] = {
        "lr": {"values": [5e-5, 1e-4, 2e-4, 5e-4]},
        "ent_coef": {"values": [0.0, 0.005, 0.01, 0.02]},
        "clip_eps": {"values": [0.15, 0.2, 0.25]},
        "gamma": {"values": [0.99, 0.995, 0.999]},
        "gae_lambda": {"values": [0.95, 0.99]},
        "update_epochs": {"values": [1, 2, 4]},
        "num_minibatches": {
            "values": [
                _divide_or_raise_error(NUM_ENVIRONMENTS_PER_BATCH, 8),  # 64 / 8 = 8
                _divide_or_raise_error(NUM_ENVIRONMENTS_PER_BATCH, 4),  # 64 / 4 = 16
                _divide_or_raise_error(NUM_ENVIRONMENTS_PER_BATCH, 2),  # 64 / 2 = 32
            ]
        },
        "obs_emb_dim": {"values": [128, 256]},
        "past_context_length": {"values": [64, 128]},
        "num_transformer_blocks": {"values": [1, 2]},
        "transformer_hidden_states_dim": {"values": [64, 128, 192]},
        "head_hidden_dim": {"values": [64, 128, 256]},
        "inject_alpha_at_trunk": {"values": [True, False]},
    }

    if training_mode == "curriculum":
        parameters.update(
            {
                "curriculum.score_lambda": {"values": [0.0, 0.25, 0.5, 0.75, 1.0]},
                "curriculum.predictor_lr": {"values": [5e-5, 1e-4, 2e-4]},
                "curriculum.lp_norm_ema_beta": {"values": [0.02, 0.05, 0.1]},
                "rnd.predictor_network_lr": {"values": [5e-5, 1e-4, 2e-4]},
                "rnd.output_embedding_dim": {"values": [128, 256]},
                "rnd.head_hidden_dim": {"values": [128, 256]},
            }
        )

    return {
        "method": "random",
        "metric": {"name": objective_metric, "goal": "maximize"},
        "parameters": parameters,
    }


def _divide_or_raise_error(numerator: int, denominator: int) -> int:
    """Helper for validating that num_envs_per_batch is divisible by num_minibatches."""
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}.")
    return numerator // denominator


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for creating or running the random-search sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="Number of random-search trials to run.")
    parser.add_argument("--project", type=str, default="open_end_proj", help="W&B project name.")
    parser.add_argument("--entity", type=str, default=None, help="Optional W&B entity/team.")
    parser.add_argument("--group", type=str, default=None, help="Optional W&B group shared across trials.")
    parser.add_argument(
        "--training-mode",
        choices=("curriculum", "baseline"),
        default="curriculum",
        help="Training pipeline to tune.",
    )
    parser.add_argument("--train-seed", type=int, default=0, help="Base seed used for all trials.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total environment steps per trial.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Existing W&B sweep id. If omitted, the script creates a new random sweep.",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Create the sweep and print its id without launching an agent.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Persist the full checkpoint/result artifact for each trial.",
    )
    return parser.parse_args()


def build_base_tuning_config(
    *,
    training_mode: str,
    entity: str | None,
    group: str | None,
    train_seed: int,
    total_timesteps: int,
    project: str = "openendedness-2026",
) -> TrainConfig:
    """Build the fixed base training config used as the starting point for every trial."""
    if training_mode == "baseline":
        return TrainConfig(
            train_seed=train_seed,
            total_timesteps=total_timesteps,
            env_id="Craftax-Classic-Symbolic-v1",
            achievement_ids_to_block=CRAFTAX_CLASSIC_INTERMEDIATE_ACHIEVEMENT_IDS,
            training_mode="baseline",
            remove_health_reward=True,
            selected_intrinsic_modules=(),
            baseline_fixed_training_alpha=(1.0,),
            num_envs_per_batch=64,
            num_steps_per_env=512,
            num_steps_per_update=256,
            eval_every_n_batches=1,
            eval_num_envs=64,
            eval_num_episodes=2,
            evaluation_alphas=((1.0,),),
            update_epochs=1,
            num_minibatches=16,
            obs_emb_dim=256,
            past_context_length=64,
            subsequence_length_in_loss_calculation=64,
            num_attn_heads=4,
            num_transformer_blocks=1,
            transformer_hidden_states_dim=64,
            qkv_features=64,
            head_hidden_dim=64,
            enable_wandb=False,
            wandb_project=project,
            wandb_entity=entity,
            wandb_group=group,
            is_timing_run=False,
        )

    if training_mode == "curriculum":
        return TrainConfig(
            train_seed=train_seed,
            total_timesteps=total_timesteps,
            env_id="Craftax-Classic-Symbolic-v1",
            achievement_ids_to_block=CRAFTAX_CLASSIC_INTERMEDIATE_ACHIEVEMENT_IDS,
            remove_health_reward=True,
            training_mode="curriculum",
            selected_intrinsic_modules=("rnd",),
            num_envs_per_batch=64,
            num_steps_per_env=512,
            num_steps_per_update=256,
            eval_every_n_batches=1,
            eval_num_envs=64,
            eval_num_episodes=2,
            evaluation_alphas=((0.8, 0.2), (1.0, 0.0)),
            update_epochs=1,
            num_minibatches=16,
            obs_emb_dim=256,
            past_context_length=64,
            subsequence_length_in_loss_calculation=64,
            num_attn_heads=4,
            num_transformer_blocks=1,
            transformer_hidden_states_dim=64,
            qkv_features=64,
            head_hidden_dim=64,
            enable_wandb=False,
            wandb_project=project,
            wandb_entity=entity,
            wandb_group=group,
            is_timing_run=False,
        )

    msg = f"Unsupported training_mode {training_mode!r}."
    raise ValueError(msg)


def build_trial_config_from_overrides(
    base_config: TrainConfig,
    overrides: Mapping[str, Any],
) -> TrainConfig:
    """Apply sampled sweep overrides to a base config and return a revalidated TrainConfig."""
    mutable_config = _clone_train_config(base_config)
    applied_override_keys: set[str] = set()

    for key, value in overrides.items():
        if key.startswith("_"):
            continue
        _set_nested_attr(mutable_config, key, value)
        applied_override_keys.add(key)

    if "transformer_hidden_states_dim" in applied_override_keys and "qkv_features" not in applied_override_keys:
        mutable_config.qkv_features = mutable_config.transformer_hidden_states_dim

    return TrainConfig(**cast(Any, _dataclass_init_kwargs(mutable_config)))


def extract_trial_summary(train_info: dict[str, Any]) -> dict[str, Any]:
    """Extract the final scalar metrics logged back to the W&B sweep run."""
    metrics = train_info["metrics"]
    eval_returns = _last_metric_slice(metrics["eval/returns"], dtype=np.float32)
    eval_lengths = _last_metric_slice(metrics["eval/lengths"], dtype=np.float32)
    eval_achievements = _last_metric_slice(metrics["eval/achievements"], dtype=np.float32)

    summary: dict[str, Any] = {
        DEFAULT_OBJECTIVE_METRIC: float(eval_returns.mean()),
        "tuning/objective_eval_return_alpha0_mean": float(eval_returns[0].mean()),
        "tuning/final_eval_length_mean": float(eval_lengths.mean()),
        "tuning/final_eval_num_achievements_mean": float(eval_achievements.sum(axis=-1).mean()),
        "tuning/final_total_env_steps": int(_last_scalar_metric(metrics["run/total_env_steps"])),
        "tuning/final_total_loss": float(_last_scalar_metric(metrics["ppo/total_loss"])),
        "tuning/final_actor_loss": float(_last_scalar_metric(metrics["ppo/actor_loss"])),
        "tuning/final_entropy": float(_last_scalar_metric(metrics["ppo/entropy"])),
        "tuning/final_approx_kl": float(_last_scalar_metric(metrics["ppo/approx_kl"])),
    }

    if "curriculum/score_mean" in metrics:
        summary["tuning/final_curriculum_score_mean"] = float(_last_scalar_metric(metrics["curriculum/score_mean"]))
    if "intrinsic_modules/rnd/predictor_loss" in metrics:
        summary["tuning/final_rnd_predictor_loss"] = float(
            _last_scalar_metric(metrics["intrinsic_modules/rnd/predictor_loss"])
        )

    return summary


def _last_metric_slice(metric_value: Any, *, dtype: Any) -> np.ndarray:
    """Return the last recorded slice from a metric history as a NumPy array."""
    metric_array = np.asarray(metric_value, dtype=dtype)
    return metric_array[-1] if metric_array.ndim > 0 else metric_array


def _last_scalar_metric(metric_value: Any) -> float | int:
    """Return the final recorded value from a scalar metric history."""
    metric_array = np.asarray(metric_value)
    return metric_array.item() if metric_array.ndim == 0 else metric_array.reshape(-1)[-1].item()


def _dataclass_init_kwargs(instance: Any) -> dict[str, Any]:
    """Return a deep-copied mapping of init-enabled dataclass fields."""
    return {
        field.name: copy.deepcopy(getattr(instance, field.name)) for field in dataclasses.fields(instance) if field.init
    }


def _clone_train_config(config: TrainConfig) -> TrainConfig:
    """Clone a TrainConfig by rebuilding it from its constructor fields."""
    return TrainConfig(**cast(Any, _dataclass_init_kwargs(config)))


def _derive_trial_seed(base_seed: int, run_id: str) -> int:
    """Derive a deterministic per-trial seed from the base seed and W&B run id."""
    seed_material = f"{base_seed}:{run_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(seed_material).digest()[:4], byteorder="big", signed=False)


def _set_nested_attr(instance: Any, dotted_path: str, value: Any) -> None:
    """Set an attribute on an object, supporting dotted paths for nested config fields."""
    path_parts = dotted_path.split(".")
    target = instance
    for path_part in path_parts[:-1]:
        if not hasattr(target, path_part):
            raise KeyError(f"Unsupported config path {dotted_path!r}.")
        target = getattr(target, path_part)

    final_part = path_parts[-1]
    if not hasattr(target, final_part):
        raise KeyError(f"Unsupported config path {dotted_path!r}.")
    setattr(target, final_part, value)


def _serialize_for_wandb(value: Any) -> Any:
    """Convert dataclass-backed config values into W&B-friendly plain Python containers."""
    if dataclasses.is_dataclass(value):
        return {
            field.name: _serialize_for_wandb(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.init
        }
    if isinstance(value, tuple):
        return [_serialize_for_wandb(item) for item in value]
    if isinstance(value, list):
        return [_serialize_for_wandb(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_for_wandb(item) for key, item in value.items()}
    return value


def _run_single_trial(base_config: TrainConfig, save_results: bool) -> None:
    """Execute one sweep trial from sampled W&B overrides through final summary logging."""
    run = wandb.init()
    if run is None:
        raise RuntimeError("wandb.init() did not return a run.")

    try:
        overrides = dict(run.config)
        overrides.setdefault("train_seed", _derive_trial_seed(base_config.train_seed, run.id))

        trial_config = build_trial_config_from_overrides(base_config=base_config, overrides=overrides)
        run.config.update(
            {
                "train_seed": trial_config.train_seed,
                "resolved_config": _serialize_for_wandb(trial_config),
            },
            allow_val_change=True,
        )
        run.name = build_wandb_run_name(trial_config)

        train_info = run_main_algo_training(config=trial_config, save_results=save_results)
        summary = extract_trial_summary(train_info=train_info)
        summary["tuning/status"] = "completed"

        wandb.log(summary)
        run.summary.update(summary)
    except Exception as exc:
        run.summary["tuning/status"] = "failed"
        run.summary["tuning/error"] = repr(exc)
        raise
    finally:
        run.finish()


if __name__ == "__main__":
    main()
