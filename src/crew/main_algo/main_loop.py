import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.train_state import TrainState
from jax.tree_util import Partial

from crew.main_algo.curriculum.alpha_sampling import sample_alpha_batch
from crew.main_algo.curriculum.lp_estimation import estimate_lp_per_reward_function
from crew.main_algo.curriculum.replay_buffer import add_alpha_score_batch
from crew.main_algo.curriculum.score_estimation import compute_scores
from crew.main_algo.curriculum.score_predictor import train_score_predictor_on_buffer
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
from crew.main_algo.types import (
    CurriculumState,
    IntrinsicStates,
    RewardNormalizationStats,
    RunnerStateTransformer,
)
from crew.main_algo.config import TrainConfig


def train_one_iteration(
    rng: jax.Array,
    agent_train_state: TrainState,
    reward_normalization_stats: RewardNormalizationStats,
    intrinsic_states: IntrinsicStates,
    curriculum_state: CurriculumState,
    env: Any,
    env_params: Any,
    intrinsic_modules: tuple[IntrinsicModule, ...],
    config: TrainConfig,
) -> tuple[jax.Array, TrainState, RewardNormalizationStats, IntrinsicStates, CurriculumState, dict[str, jax.Array]]:
    """Run one outer iteration: alpha sampling, inner updates, then intrinsic updates."""
    rng, alpha_batch, alpha_sampling_metrics = sample_alpha_batch(
        rng=rng,
        curriculum_state=curriculum_state,
        config=config,
    )
    ########### Sample new envs and reset memory ############
    rng, reset_rng_base = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng_base, num=config.num_envs_per_batch)
    prev_obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
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
    reward_normalization_stats = reward_normalization_stats.replace(
        running_forward_return=jnp.zeros_like(reward_normalization_stats.running_forward_return),
    )

    ############ Data collection and agent training ############

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

    (
        runner_state,
        (
            inner_metrics,
            intrinsic_modules_update_data,
            lp_estimation_data,
        ),
    ) = jax.lax.scan(
        Partial(
            collect_data_and_update_agent,
            env=env,
            env_params=env_params,
            alpha_batch=alpha_batch,
            intrinsic_states=intrinsic_states,
            intrinsic_modules=intrinsic_modules,
            config=config,
        ),
        runner_state,
        None,
        config.num_updates_per_batch,
    )
    rng = runner_state.rng
    agent_train_state = runner_state.agent_train_state
    reward_normalization_stats = runner_state.reward_normalization_stats
    # Average each metric over the U inner agent updates.
    inner_metrics = jtu.tree_map(lambda x: jnp.mean(x, axis=0), inner_metrics)

    ############### Intrinsic reward modules training ############

    # Collapse [U, T, B, ...] to [U*T, B, ...] before intrinsic module updates.
    intrinsic_modules_update_data = jtu.tree_map(
        lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])),
        intrinsic_modules_update_data,
    )
    rng, intrinsic_states, intrinsic_update_metrics = update_intrinsic_modules(
        rng=rng,
        intrinsic_modules=intrinsic_modules,
        intrinsic_states=intrinsic_states,
        intrinsic_modules_update_data=intrinsic_modules_update_data,
        config=config,
    )

    ############### LP estimation, score computation, and buffer updates ############
    lp_per_reward_function, single_episode_env_mask = estimate_lp_per_reward_function(
        lp_estimation_data=lp_estimation_data,
        reward_normalization_stats=reward_normalization_stats,
        is_episodic_per_reward_function=config.is_episodic_per_reward_function,
        gamma_per_reward_function=config.gamma_per_reward_function,
        eps=config.reward_norm_eps,
    )
    scores, score_metrics = compute_scores(
        alpha_batch=alpha_batch,
        lp_per_reward_function=lp_per_reward_function,
        score_lp_mode=config.curriculum.score_lp_mode,
        score_lambda=config.curriculum.score_lambda,
    )

    # Envs with only one completed episode have non-valid score targets.
    score_datapoint_is_valid = jnp.logical_not(single_episode_env_mask.astype(jnp.bool_))
    alpha_score_replay_buffer = add_alpha_score_batch(
        alpha_score_replay_buffer=curriculum_state.alpha_score_replay_buffer,
        alpha_batch=alpha_batch,
        score_batch=scores,
        is_valid_batch=score_datapoint_is_valid,
    )
    curriculum_state = curriculum_state.replace(
        alpha_score_replay_buffer=alpha_score_replay_buffer,
    )
    rng, score_predictor_train_state, predictor_metrics = train_score_predictor_on_buffer(
        rng=rng,
        score_predictor_train_state=curriculum_state.score_predictor_train_state,
        alpha_score_replay_buffer=alpha_score_replay_buffer,
        config=config,
    )
    curriculum_state = curriculum_state.replace(
        score_predictor_train_state=score_predictor_train_state,
        num_batches_seen=curriculum_state.num_batches_seen
        + jnp.array(1, dtype=curriculum_state.num_batches_seen.dtype),
    )

    ############### Metrics merging and aggregation ############
    completed_episodes_per_env = jnp.sum(
        lp_estimation_data.done_masks[..., 0].astype(jnp.float32),
        axis=(0, 1),
    )  # [B]
    alpha_entropy_per_env = -jnp.sum(
        alpha_batch * jnp.log(jnp.clip(alpha_batch, a_min=1e-8)),
        axis=1,
    )  # [B]

    metrics = (
        inner_metrics
        | intrinsic_update_metrics
        | alpha_sampling_metrics
        | score_metrics
        | predictor_metrics
        | {
            "curriculum/alpha/mean_per_reward_function": jnp.mean(alpha_batch, axis=0),  # [R]
            "curriculum/alpha/std_per_reward_function": jnp.std(alpha_batch, axis=0),  # [R]
            "curriculum/alpha/entropy_mean": jnp.mean(alpha_entropy_per_env),  # scalar
            "curriculum/lp_per_reward_function": jnp.mean(lp_per_reward_function, axis=0),  # [R]
            "curriculum/valid_fraction_of_scores_in_batch": jnp.mean(
                score_datapoint_is_valid.astype(jnp.float32)
            ),  # scalar
            "curriculum/completed_episodes_per_env_mean": jnp.mean(completed_episodes_per_env),  # scalar
            "curriculum/alpha/extrinsic_weight_per_env": alpha_batch[:, 0],  # [B]
        }
    )
    return rng, agent_train_state, reward_normalization_stats, intrinsic_states, curriculum_state, metrics


def full_training(
    rng: jax.Array,
    agent_train_state: TrainState,
    reward_normalization_stats: RewardNormalizationStats,
    intrinsic_states: IntrinsicStates,
    curriculum_state: CurriculumState,
    env: Any,
    env_params: Any,
    intrinsic_modules: tuple[IntrinsicModule, ...],
    config: TrainConfig,
) -> dict[str, Any]:
    """Main training loop with fixed-alpha windows and intrinsic updates."""
    achievement_names = infer_achievement_names(env=env, env_params=env_params)
    wandb_run = init_wandb_run(config=config)

    train_one_iteration_jit = jax.jit(
        Partial(
            train_one_iteration,
            env=env,
            env_params=env_params,
            intrinsic_modules=intrinsic_modules,
            config=config,
        )
    )
    evaluate_policy_on_alphas_jit = jax.jit(
        Partial(
            evaluate_policy_on_alphas,
            env=env,
            env_params=env_params,
            evaluation_alphas=config.evaluation_alphas_array,
            num_eval_envs=config.eval_num_envs,
            num_eval_episodes=config.eval_num_episodes,
            achievement_names=achievement_names,
            config=config,
        )
    )

    metrics_per_batch = []
    run_time_metrics_per_batch = []
    eval_metrics_per_call = []
    env_steps_per_batch = config.num_envs_per_batch * config.num_steps_per_env
    wall_clock_start = time.perf_counter()
    previous_iteration_end_time = wall_clock_start

    for batch_idx in range(config.num_batches_of_envs):
        if config.is_timing_run:
            train_iteration_start_time = time.perf_counter()
        rng, agent_train_state, reward_normalization_stats, intrinsic_states, curriculum_state, metrics = (
            train_one_iteration_jit(
                rng,
                agent_train_state,
                reward_normalization_stats,
                intrinsic_states,
                curriculum_state,
            )
        )
        if config.is_timing_run:
            metrics = jax.block_until_ready(metrics)
            train_iteration_time_sec = time.perf_counter() - train_iteration_start_time
            print(f"[timing] batch {batch_idx + 1}: " f"train_one_iteration={train_iteration_time_sec:.6f}s")

        metrics_per_batch.append(metrics)
        batch_num = batch_idx + 1
        total_env_steps = batch_num * env_steps_per_batch

        now = time.perf_counter()
        batch_wall_clock_sec = max(now - previous_iteration_end_time, 1e-8)
        cumulative_wall_clock_sec = now - wall_clock_start
        previous_iteration_end_time = now

        run_time_metrics = {
            "run/batch_idx": jnp.array(batch_num, dtype=jnp.int32),
            "run/total_env_steps": jnp.array(total_env_steps, dtype=jnp.int32),
            "time/cumulative_wall_clock_sec": jnp.array(cumulative_wall_clock_sec, dtype=jnp.float32),
            "time/env_steps_per_sec": jnp.array(env_steps_per_batch / batch_wall_clock_sec, dtype=jnp.float32),
        }
        run_time_metrics_per_batch.append(run_time_metrics)

        eval_metrics_for_logging = None
        if batch_idx % config.eval_every_n_batches == 0:
            if config.is_timing_run:
                eval_start_time = time.perf_counter()
            rng, eval_metrics = evaluate_policy_on_alphas_jit(
                rng,
                agent_train_state,
            )
            if config.is_timing_run:
                eval_metrics = jax.block_until_ready(eval_metrics)
                eval_time_sec = time.perf_counter() - eval_start_time
                print(f"[timing] batch {batch_num}: " f"evaluation={eval_time_sec:.6f}s")
            eval_metrics = eval_metrics | {
                "eval/batch_idx": jnp.array(batch_num, dtype=jnp.int32),
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
                print(f"[timing] batch {batch_num}: " f"logging={logging_time_sec:.6f}s")
        except Exception as error:  # pragma: no cover - logging must not stop training
            print(f"Failed to log outer batch to Weights & Biases: {error}")

    finish_wandb_run(wandb_run)

    metrics = jtu.tree_map(
        lambda *x: jnp.stack(x, axis=0),
        *metrics_per_batch,
    )
    run_time_metrics = jtu.tree_map(
        lambda *x: jnp.stack(x, axis=0),
        *run_time_metrics_per_batch,
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
        "rng": rng,
        "agent_state": agent_train_state,
        "reward_normalization_stats": reward_normalization_stats,
        "intrinsic_states": intrinsic_states,
        "curriculum_state": curriculum_state,
        "metrics": metrics,
    }
