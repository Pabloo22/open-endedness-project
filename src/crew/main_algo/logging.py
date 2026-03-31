"""Weights & Biases logging utilities for main_algo outer-loop metrics."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np

import wandb
from crew.experiments.identity import build_experiment_identity


def build_reward_function_names(config: Any) -> tuple[str, ...]:
    """Return reward function names in metric vector order."""
    return ("extrinsic", *tuple(config.selected_intrinsic_modules))


def build_wandb_run_name(config: Any) -> str:
    """Build deterministic default run name when no explicit name is provided."""
    if config.wandb_run_name is not None:
        return config.wandb_run_name
    return build_experiment_identity(config).run_name


def build_wandb_group(config: Any) -> str:
    """Build deterministic default run group when no explicit group is provided."""
    if config.wandb_group is not None:
        return config.wandb_group
    return build_experiment_identity(config).run_group


def build_wandb_tags(config: Any) -> tuple[str, ...]:
    """Build stable default tags merged with user-provided tags."""
    identity = build_experiment_identity(config)
    return tuple(dict.fromkeys((*identity.tags, *tuple(config.wandb_tags))))


def init_wandb_run(config: Any) -> Any | None:
    """Initialize W&B run when enabled in config."""
    if not config.enable_wandb:
        return None

    # If W&B is already initialized (e.g. by a sweep agent), just reuse it.
    if wandb.run is not None:
        return {"sweep_run": wandb.run}

    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        group=build_wandb_group(config),
        name=build_wandb_run_name(config),
        tags=list(build_wandb_tags(config)),
        config=dataclasses.asdict(config),
    )


def finish_wandb_run(run: Any | None) -> None:
    """Finish W&B run if present."""
    if run is None:
        return

    # If this run was passed down from a sweep agent, let the agent finish it.
    if isinstance(run, dict) and "sweep_run" in run:
        return

    run.finish()


def _split_reward_vector_metric(
    payload: dict[str, int | float],
    metric_key: str,
    metric_value: Any,
    reward_function_names: tuple[str, ...],
) -> None:
    metric_value = jnp.asarray(metric_value)
    for reward_name, reward_value in zip(reward_function_names, metric_value):
        payload[f"{metric_key}/{reward_name}"] = float(reward_value)


def _append_intrinsic_module_scalar_metrics(
    payload: dict[str, int | float],
    batch_metrics: dict[str, Any],
) -> None:
    """Append scalar intrinsic-module metrics present in `batch_metrics`."""
    for key in sorted(batch_metrics):
        if not key.startswith("intrinsic_modules/"):
            continue
        metric_value = jnp.asarray(batch_metrics[key])
        if metric_value.ndim == 0:
            payload[key] = metric_value.item()


def _build_training_batch_log_payload_curriculum(
    batch_metrics: dict[str, Any],
    reward_function_names: tuple[str, ...],
) -> dict[str, int | float]:
    """Build logging payload for one training batch from saved metrics."""
    scalar_metric_keys = (
        "run/batch_idx",
        "run/total_env_steps",
        "time/cumulative_wall_clock_sec",
        "time/env_steps_per_sec",
        "preproc/weighted_adv_mean",
        "preproc/weighted_adv_std",
        "ppo/total_loss",
        "ppo/actor_loss",
        "ppo/entropy",
        "ppo/approx_kl",
        "curriculum/pred_score_mean",
        "curriculum/predictor_loss",
        "curriculum/alpha/entropy_mean",
        "curriculum/score_mean",
        "curriculum/valid_fraction_of_scores_in_batch",
        "curriculum/completed_episodes_per_env_mean",
    )
    reward_vector_metric_keys = (
        "preproc/adv_raw_mean",
        "preproc/adv_norm_mean",
        "preproc/adv_norm_std",
        "ppo/value_loss",
        "curriculum/alpha/mean_per_reward_function",
        "curriculum/alpha/std_per_reward_function",
        "curriculum/lp_per_reward_function",
    )

    payload: dict[str, int | float] = {}
    for scalar_key in scalar_metric_keys:
        payload[scalar_key] = jnp.asarray(batch_metrics[scalar_key]).item()
    _append_intrinsic_module_scalar_metrics(payload=payload, batch_metrics=batch_metrics)

    for reward_metric_key in reward_vector_metric_keys:
        _split_reward_vector_metric(
            payload=payload,
            metric_key=reward_metric_key,
            metric_value=batch_metrics[reward_metric_key],
            reward_function_names=reward_function_names,
        )
    return payload


def _build_training_batch_log_payload_baseline(
    batch_metrics: dict[str, Any],
    reward_function_names: tuple[str, ...],
) -> dict[str, int | float]:
    scalar_metric_keys = (
        "run/batch_idx",
        "run/total_env_steps",
        "time/cumulative_wall_clock_sec",
        "time/env_steps_per_sec",
        "preproc/weighted_adv_mean",
        "preproc/weighted_adv_std",
        "ppo/total_loss",
        "ppo/actor_loss",
        "ppo/entropy",
        "ppo/approx_kl",
    )
    reward_vector_metric_keys = (
        "preproc/adv_raw_mean",
        "preproc/adv_norm_mean",
        "preproc/adv_norm_std",
        "ppo/value_loss",
    )

    payload: dict[str, int | float] = {}
    for scalar_key in scalar_metric_keys:
        payload[scalar_key] = jnp.asarray(batch_metrics[scalar_key]).item()
    _append_intrinsic_module_scalar_metrics(payload=payload, batch_metrics=batch_metrics)

    for reward_metric_key in reward_vector_metric_keys:
        _split_reward_vector_metric(
            payload=payload,
            metric_key=reward_metric_key,
            metric_value=batch_metrics[reward_metric_key],
            reward_function_names=reward_function_names,
        )
    return payload


def build_training_batch_log_payload(
    batch_metrics: dict[str, Any],
    reward_function_names: tuple[str, ...],
    training_mode: str = "curriculum",
) -> dict[str, int | float]:
    if training_mode == "baseline":
        return _build_training_batch_log_payload_baseline(
            batch_metrics=batch_metrics,
            reward_function_names=reward_function_names,
        )
    return _build_training_batch_log_payload_curriculum(
        batch_metrics=batch_metrics,
        reward_function_names=reward_function_names,
    )


def _build_eval_log_payload_curriculum(
    eval_metrics: dict[str, Any],
    evaluation_alpha_labels: tuple[str, ...],
    achievement_names: tuple[str, ...],
) -> dict[str, int | float]:
    """Build eval logging payload for curriculum mode."""
    returns = jnp.asarray(eval_metrics["eval/returns"])  # [A, B, E]
    lengths = jnp.asarray(eval_metrics["eval/lengths"])  # [A, B, E]
    achievements = jnp.asarray(eval_metrics["eval/achievements"])  # [A, B, E, K]

    payload: dict[str, int | float] = {
        "eval/batch_num": jnp.asarray(eval_metrics["eval/batch_idx"]).item(),
        "eval/total_steps": jnp.asarray(eval_metrics["eval/total_steps"]).item(),
    }
    for alpha_idx, alpha_label in enumerate(evaluation_alpha_labels):
        alpha_returns = returns[alpha_idx]
        alpha_lengths = lengths[alpha_idx]
        alpha_achievements = achievements[alpha_idx].astype(jnp.float32)

        payload[f"eval/{alpha_label}/return_mean"] = float(jnp.mean(alpha_returns))
        payload[f"eval/{alpha_label}/return_std"] = float(jnp.std(alpha_returns))
        payload[f"eval/{alpha_label}/return_median"] = float(jnp.median(alpha_returns))
        payload[f"eval/{alpha_label}/length_mean"] = float(jnp.mean(alpha_lengths))
        payload[f"eval/{alpha_label}/length_std"] = float(jnp.std(alpha_lengths))
        payload[f"eval/{alpha_label}/length_median"] = float(jnp.median(alpha_lengths))
        payload[f"eval/{alpha_label}/num_accomplished_achievements_mean"] = float(
            jnp.mean(jnp.sum(alpha_achievements, axis=-1))
        )

        for achievement_idx, achievement_name in enumerate(achievement_names):
            achievement_success_percentage = jnp.mean(alpha_achievements[..., achievement_idx]) * 100.0
            payload[f"eval/{alpha_label}/achievement_success_percentage/{achievement_name}"] = float(
                achievement_success_percentage
            )
    return payload


def _build_eval_log_payload_baseline(
    eval_metrics: dict[str, Any],
    achievement_names: tuple[str, ...],
) -> dict[str, int | float]:
    """Build eval logging payload for baseline mode (single fixed alpha, no alpha-label names)."""
    returns = jnp.asarray(eval_metrics["eval/returns"])[0]  # [B, E]
    lengths = jnp.asarray(eval_metrics["eval/lengths"])[0]  # [B, E]
    achievements = jnp.asarray(eval_metrics["eval/achievements"])[0].astype(jnp.float32)  # [B, E, K]

    payload: dict[str, int | float] = {
        "eval/batch_num": jnp.asarray(eval_metrics["eval/batch_idx"]).item(),
        "eval/total_steps": jnp.asarray(eval_metrics["eval/total_steps"]).item(),
        "eval/return_mean": float(jnp.mean(returns)),
        "eval/return_std": float(jnp.std(returns)),
        "eval/return_median": float(jnp.median(returns)),
        "eval/length_mean": float(jnp.mean(lengths)),
        "eval/length_std": float(jnp.std(lengths)),
        "eval/length_median": float(jnp.median(lengths)),
        "eval/num_accomplished_achievements_mean": float(jnp.mean(jnp.sum(achievements, axis=-1))),
    }
    for achievement_idx, achievement_name in enumerate(achievement_names):
        achievement_success_percentage = jnp.mean(achievements[..., achievement_idx]) * 100.0
        payload[f"eval/achievement_success_percentage/{achievement_name}"] = float(achievement_success_percentage)
    return payload


def build_eval_log_payload(
    eval_metrics: dict[str, Any],
    evaluation_alpha_labels: tuple[str, ...],
    achievement_names: tuple[str, ...],
    training_mode: str = "curriculum",
) -> dict[str, int | float]:
    """Build eval logging payload for one eval call from saved eval slice."""
    if training_mode == "baseline":
        return _build_eval_log_payload_baseline(
            eval_metrics=eval_metrics,
            achievement_names=achievement_names,
        )
    return _build_eval_log_payload_curriculum(
        eval_metrics=eval_metrics,
        evaluation_alpha_labels=evaluation_alpha_labels,
        achievement_names=achievement_names,
    )


def log_outer_batch_to_wandb(
    run: Any | None,
    batch_metrics: dict[str, Any],
    config: Any,
    eval_metrics: dict[str, Any] | None = None,
    achievement_names: tuple[str, ...] | None = None,
) -> None:
    """Log one outer training batch (and optional eval slice) to W&B."""
    if run is None:
        return

    reward_function_names = build_reward_function_names(config)
    payload = build_training_batch_log_payload(
        batch_metrics=batch_metrics,
        reward_function_names=reward_function_names,
        training_mode=config.training_mode,
    )
    if config.training_mode == "curriculum":
        payload["curriculum/alpha/extrinsic_weight_histogram"] = wandb.Histogram(
            np.asarray(batch_metrics["curriculum/alpha/extrinsic_weight_per_env"])
        )
    if eval_metrics is not None:
        payload = payload | build_eval_log_payload(
            eval_metrics=eval_metrics,
            evaluation_alpha_labels=config.evaluation_alpha_labels,
            achievement_names=achievement_names if achievement_names is not None else tuple(),
            training_mode=config.training_mode,
        )

    wandb.log(
        payload,
        step=int(jnp.asarray(batch_metrics["run/total_env_steps"]).item()),
        commit=True,
    )
