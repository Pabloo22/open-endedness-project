"""Random-search hyperparameter tuning entrypoint backed by W&B sweeps.

This module creates a W&B sweep with method="random" and runs trials through
the existing training entrypoint. Each sweep trial owns a single W&B run. The
inner training loop's W&B logging is disabled to avoid nested runs, and the
trial logs a compact final summary for ranking and comparison.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import wandb

from crew.experiments.run_training import run_main_algo_training
from crew.main_algo.config import TrainConfig
from crew.main_algo.logging import build_wandb_run_name


DEFAULT_OBJECTIVE_METRIC = "tuning/objective_eval_return_mean"


def _dataclass_init_kwargs(instance: Any) -> dict[str, Any]:
    return {
        field.name: copy.deepcopy(getattr(instance, field.name)) for field in dataclasses.fields(instance) if field.init
    }


def _clone_train_config(config: TrainConfig) -> TrainConfig:
    return TrainConfig(**cast(Any, _dataclass_init_kwargs(config)))


def _set_nested_attr(instance: Any, dotted_path: str, value: Any) -> None:
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


def build_base_tuning_config(
    *,
    training_mode: str,
    project: str,
    entity: str | None,
    group: str | None,
    train_seed: int,
    total_timesteps: int,
) -> TrainConfig:
    if training_mode == "baseline":
        return TrainConfig(
            train_seed=train_seed,
            total_timesteps=total_timesteps,
            env_id="Craftax-Classic-Symbolic-v1",
            achievement_ids_to_block=tuple(range(15)),
            training_mode="baseline",
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
            achievement_ids_to_block=tuple(range(15)),
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


def build_default_sweep_config(
    *,
    training_mode: str,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "lr": {"values": [5e-5, 1e-4, 2e-4, 5e-4]},
        "ent_coef": {"values": [0.0, 0.005, 0.01, 0.02]},
        "clip_eps": {"values": [0.1, 0.2, 0.3]},
        "gamma": {"values": [0.99, 0.995]},
        "gae_lambda": {"values": [0.95, 0.99]},
        "update_epochs": {"values": [1, 2, 4]},
        "num_minibatches": {"values": [8, 16, 32]},
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


def build_trial_config_from_overrides(
    base_config: TrainConfig,
    overrides: Mapping[str, Any],
) -> TrainConfig:
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
    metrics = train_info["metrics"]
    eval_returns = np.asarray(metrics["eval/returns"], dtype=np.float32)
    eval_lengths = np.asarray(metrics["eval/lengths"], dtype=np.float32)
    eval_achievements = np.asarray(metrics["eval/achievements"], dtype=np.float32)

    summary: dict[str, Any] = {
        DEFAULT_OBJECTIVE_METRIC: float(eval_returns.mean()),
        "tuning/objective_eval_return_alpha0_mean": float(eval_returns[0].mean()),
        "tuning/final_eval_length_mean": float(eval_lengths.mean()),
        "tuning/final_eval_num_achievements_mean": float(eval_achievements.sum(axis=-1).mean()),
        "tuning/final_total_env_steps": int(np.asarray(metrics["run/total_env_steps"]).item()),
        "tuning/final_total_loss": float(np.asarray(metrics["ppo/total_loss"]).item()),
        "tuning/final_actor_loss": float(np.asarray(metrics["ppo/actor_loss"]).item()),
        "tuning/final_entropy": float(np.asarray(metrics["ppo/entropy"]).item()),
        "tuning/final_approx_kl": float(np.asarray(metrics["ppo/approx_kl"]).item()),
    }

    if "curriculum/score_mean" in metrics:
        summary["tuning/final_curriculum_score_mean"] = float(np.asarray(metrics["curriculum/score_mean"]).item())
    if "intrinsic_modules/rnd/predictor_loss" in metrics:
        summary["tuning/final_rnd_predictor_loss"] = float(
            np.asarray(metrics["intrinsic_modules/rnd/predictor_loss"]).item()
        )

    return summary


def _run_single_trial(base_config: TrainConfig, save_results: bool) -> None:
    run = wandb.init()
    if run is None:
        raise RuntimeError("wandb.init() did not return a run.")

    try:
        trial_config = build_trial_config_from_overrides(base_config=base_config, overrides=dict(run.config))
        run.config.update({"resolved_config": _serialize_for_wandb(trial_config)}, allow_val_change=True)
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


def parse_args() -> argparse.Namespace:
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


def main() -> None:
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


if __name__ == "__main__":
    main()
