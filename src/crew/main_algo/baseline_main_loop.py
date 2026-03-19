import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.tree_util import Partial

from crew.main_algo.data_collection_and_agent_updates import (
    collect_data_and_update_agent,
)
from crew.main_algo.evaluation import evaluate_policy_on_alphas, infer_achievement_names
from crew.main_algo.intrinsic_modules.api import IntrinsicModule
from crew.main_algo.intrinsic_modules.update_loop import update_intrinsic_modules
from crew.main_algo.logging import (
    finish_wandb_run,
    init_wandb_run,
    log_outer_batch_to_wandb,
)
from crew.main_algo.video import record_and_log_videos
from crew.main_algo.types import (
    IntrinsicStates,
    RewardNormalizationStats,
    RunnerStateTransformer,
)
from crew.main_algo.wrappers import AutoResetEnvWrapper, OptimisticResetVecEnvWrapper


def train_one_iteration_baseline(
    runner_state: RunnerStateTransformer,
    intrinsic_states: IntrinsicStates,
    alpha_batch: jax.Array,
    env: Any,
    env_params: Any,
    intrinsic_modules: tuple[IntrinsicModule, ...],
    config: Any,
) -> tuple[RunnerStateTransformer, IntrinsicStates, dict[str, jax.Array]]:
    """Run one baseline update: collect/update agent, then update intrinsic modules."""
    (
        runner_state,
        (
            agent_update_metrics,
            intrinsic_modules_update_data,
            _,
        ),
    ) = collect_data_and_update_agent(
        runner_state=runner_state,
        _unused=None,
        env=env,
        env_params=env_params,
        alpha_batch=alpha_batch,
        intrinsic_states=intrinsic_states,
        intrinsic_modules=intrinsic_modules,
        config=config,
    )

    rng, intrinsic_states, intrinsic_update_metrics = update_intrinsic_modules(
        rng=runner_state.rng,
        intrinsic_modules=intrinsic_modules,
        intrinsic_states=intrinsic_states,
        intrinsic_modules_update_data=intrinsic_modules_update_data,
        config=config,
    )
    runner_state = runner_state.replace(rng=rng)
    metrics = agent_update_metrics | intrinsic_update_metrics
    return runner_state, intrinsic_states, metrics


def full_training_baseline(
    rng: jax.Array,
    agent_train_state: Any,
    reward_normalization_stats: RewardNormalizationStats,
    intrinsic_states: IntrinsicStates,
    env: Any,
    env_params: Any,
    intrinsic_modules: tuple[IntrinsicModule, ...],
    config: Any,
) -> dict[str, Any]:
    """Main training loop for fixed-alpha baseline mode."""
    eval_env = AutoResetEnvWrapper(env._env) if isinstance(env, OptimisticResetVecEnvWrapper) else env
    achievement_names = infer_achievement_names(env=eval_env, env_params=env_params)
    wandb_run = init_wandb_run(config=config)

    alpha = jnp.asarray(config.baseline_fixed_training_alpha, dtype=jnp.float32)
    alpha_batch = jnp.broadcast_to(alpha[None, :], (config.num_envs_per_batch, config.num_reward_functions))

    # Sample initial envs and initialize memory caches once.
    rng, reset_rng = jax.random.split(rng)
    prev_obs, env_state = env.reset(reset_rng, env_params)
    prev_done = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.bool_)

    memories = jnp.zeros(
        (
            config.num_envs_per_batch,
            config.past_context_length,
            config.num_transformer_blocks,
            config.transformer_hidden_states_dim,
        ),
        dtype=jnp.float32,
    )
    memories_mask = jnp.zeros(
        (
            config.num_envs_per_batch,
            config.num_attn_heads,
            1,
            config.past_context_length + 1,
        ),
        dtype=jnp.bool_,
    )
    memories_mask_idx = jnp.full(
        (config.num_envs_per_batch,),
        config.past_context_length + 1,
        dtype=jnp.int32,
    )

    runner_state = RunnerStateTransformer(
        rng=rng,
        agent_train_state=agent_train_state,
        env_state=env_state,
        prev_obs=prev_obs,
        prev_done=prev_done,
        reward_normalization_stats=reward_normalization_stats,
        memories=memories,
        memories_mask=memories_mask,
        memories_mask_idx=memories_mask_idx,
    )

    train_one_iteration_baseline_jit = jax.jit(
        Partial(
            train_one_iteration_baseline,
            alpha_batch=alpha_batch,
            env=env,
            env_params=env_params,
            intrinsic_modules=intrinsic_modules,
            config=config,
        )
    )
    evaluate_policy_on_alphas_jit = jax.jit(
        Partial(
            evaluate_policy_on_alphas,
            env=eval_env,
            env_params=env_params,
            evaluation_alphas=config.evaluation_alphas_array,
            num_eval_envs=config.eval_num_envs,
            num_eval_episodes=config.eval_num_episodes,
            achievement_names=achievement_names,
            config=config,
        )
    )

    metrics_per_update = []
    run_time_metrics_per_update = []
    eval_metrics_per_call = []
    env_steps_per_update = config.num_envs_per_batch * config.num_steps_per_update
    total_num_updates = config.num_batches_of_envs * config.num_updates_per_batch
    eval_every_n_updates = config.eval_every_n_batches * config.num_updates_per_batch
    wall_clock_start = time.perf_counter()
    previous_update_end_time = wall_clock_start

    for update_idx in range(total_num_updates):
        if config.is_timing_run:
            train_iteration_start_time = time.perf_counter()
        runner_state, intrinsic_states, metrics = train_one_iteration_baseline_jit(
            runner_state,
            intrinsic_states,
        )
        if config.is_timing_run:
            metrics = jax.block_until_ready(metrics)
            train_iteration_time_sec = time.perf_counter() - train_iteration_start_time
            print(f"[timing] update {update_idx + 1}: " f"train_one_iteration={train_iteration_time_sec:.6f}s")

        metrics_per_update.append(metrics)
        update_num = update_idx + 1
        total_env_steps = update_num * env_steps_per_update

        now = time.perf_counter()
        update_wall_clock_sec = max(now - previous_update_end_time, 1e-8)
        cumulative_wall_clock_sec = now - wall_clock_start
        previous_update_end_time = now

        run_time_metrics = {
            "run/batch_idx": jnp.array(update_num, dtype=jnp.int32),
            "run/total_env_steps": jnp.array(total_env_steps, dtype=jnp.int32),
            "time/cumulative_wall_clock_sec": jnp.array(cumulative_wall_clock_sec, dtype=jnp.float32),
            "time/env_steps_per_sec": jnp.array(env_steps_per_update / update_wall_clock_sec, dtype=jnp.float32),
        }
        run_time_metrics_per_update.append(run_time_metrics)

        eval_metrics_for_logging = None
        # Match curriculum eval frequency in environment steps:
        # curriculum eval happens after updates U, U+E*U, U+2*E*U, ...
        if (
            update_num >= config.num_updates_per_batch
            and (update_num - config.num_updates_per_batch) % eval_every_n_updates == 0
        ):
            if config.is_timing_run:
                eval_start_time = time.perf_counter()
            eval_rng, eval_metrics = evaluate_policy_on_alphas_jit(
                runner_state.rng,
                runner_state.agent_train_state,
            )
            runner_state = runner_state.replace(rng=eval_rng)
            if config.is_timing_run:
                eval_metrics = jax.block_until_ready(eval_metrics)
                eval_time_sec = time.perf_counter() - eval_start_time
                print(f"[timing] update {update_num}: " f"evaluation={eval_time_sec:.6f}s")
            eval_metrics = eval_metrics | {
                "eval/batch_idx": jnp.array(update_num, dtype=jnp.int32),
                "eval/total_steps": jnp.array(total_env_steps, dtype=jnp.int32),
            }
            eval_metrics_for_logging = eval_metrics
            eval_metrics_per_call.append(eval_metrics)

        try:
            if config.is_timing_run:
                logging_start_time = time.perf_counter()
            log_outer_batch_to_wandb(
                run=wandb_run,
                batch_metrics=metrics | run_time_metrics,
                config=config,
                eval_metrics=eval_metrics_for_logging,
                achievement_names=achievement_names,
            )
            if config.is_timing_run:
                logging_time_sec = time.perf_counter() - logging_start_time
                print(f"[timing] update {update_num}: " f"logging={logging_time_sec:.6f}s")
        except Exception as error:  # pragma: no cover - logging must not stop training
            print(f"Failed to log outer update to Weights & Biases: {error}")

    if config.video_num_episodes > 0:
        try:
            record_and_log_videos(
                run=wandb_run,
                train_state=runner_state.agent_train_state,
                env=eval_env,
                env_params=env_params,
                config=config,
                alpha=alpha,
                num_episodes=config.video_num_episodes,
                step=total_env_steps,
                fps=config.video_fps,
                rng=runner_state.rng,
            )
        except Exception as error:
            print(f"Failed to record/log videos: {error}")

    finish_wandb_run(wandb_run)

    metrics = jtu.tree_map(
        lambda *x: jnp.stack(x, axis=0),
        *metrics_per_update,
    )
    run_time_metrics = jtu.tree_map(
        lambda *x: jnp.stack(x, axis=0),
        *run_time_metrics_per_update,
    )
    eval_metrics = jtu.tree_map(
        lambda *x: jnp.stack(x, axis=0),
        *eval_metrics_per_call,
    )

    metrics = (
        metrics
        | run_time_metrics
        | eval_metrics
        | {
            "eval/alphas": config.evaluation_alphas_array,
            "eval/achievement_names": achievement_names,
        }
    )
    return {
        "rng": runner_state.rng,
        "agent_state": runner_state.agent_train_state,
        "reward_normalization_stats": runner_state.reward_normalization_stats,
        "intrinsic_states": intrinsic_states,
        "metrics": metrics,
    }
