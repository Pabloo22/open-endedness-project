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
from crew.main_algo.data_collection_and_agent_updates import collect_data_and_update_agent
from crew.main_algo.intrinsic_modules.api import IntrinsicModule
from crew.main_algo.intrinsic_modules.update_loop import update_intrinsic_modules
from crew.main_algo.types import CurriculumState, IntrinsicStates, RewardNormalizationStats, RunnerStateTransformer


def train_one_outer_iteration(
    rng: jax.Array,
    agent_train_state: TrainState,
    reward_normalization_stats: RewardNormalizationStats,
    intrinsic_states: IntrinsicStates,
    curriculum_state: CurriculumState,
    env: Any,
    env_params: Any,
    intrinsic_modules: tuple[IntrinsicModule, ...],
    config: Any,
) -> tuple[jax.Array, TrainState, RewardNormalizationStats, IntrinsicStates, CurriculumState, dict[str, jax.Array]]:
    """Run one outer iteration: alpha sampling, inner updates, then intrinsic updates."""
    rng, alpha_batch, alpha_sampling_diagnostics = sample_alpha_batch(
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
            inner_diagnostics,
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
    inner_diagnostics = jtu.tree_map(lambda x: jnp.mean(x, axis=0), inner_diagnostics)

    ############### Intrinsic reward modules training ############

    # Collapse [U, T, B, ...] to [U*T, B, ...] before intrinsic module updates.
    intrinsic_modules_update_data = jtu.tree_map(
        lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])),
        intrinsic_modules_update_data,
    )
    rng, intrinsic_states, intrinsic_update_diagnostics = update_intrinsic_modules(
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
    scores, score_diagnostics = compute_scores(
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
    rng, score_predictor_train_state, predictor_diagnostics = train_score_predictor_on_buffer(
        rng=rng,
        score_predictor_train_state=curriculum_state.score_predictor_train_state,
        alpha_score_replay_buffer=alpha_score_replay_buffer,
        config=config,
    )
    curriculum_state = curriculum_state.replace(
        score_predictor_train_state=score_predictor_train_state,
        num_batches_seen=curriculum_state.num_batches_seen + jnp.array(1, dtype=curriculum_state.num_batches_seen.dtype),
    )

    ############### Diagnostics aggregation ############

    diagnostics = (
        inner_diagnostics
        | intrinsic_update_diagnostics
        | alpha_sampling_diagnostics
        | score_diagnostics
        | predictor_diagnostics
        | {
            "alpha/mean_per_reward_function": jnp.mean(alpha_batch, axis=0),  # [R]
            "alpha/std_per_reward_function": jnp.std(alpha_batch, axis=0),  # [R]
            "lp/signed_mean_per_reward_function": jnp.mean(lp_per_reward_function, axis=0),  # [R]
            "lp/signed_std_per_reward_function": jnp.std(lp_per_reward_function, axis=0),  # [R]
            "lp/single_episode_env_fraction": jnp.mean(single_episode_env_mask.astype(jnp.float32)),  # scalar
            "lp/single_episode_env_count": jnp.sum(single_episode_env_mask).astype(jnp.float32),  # scalar
            "score/valid_fraction_in_batch": jnp.mean(score_datapoint_is_valid.astype(jnp.float32)),  # scalar
            "curriculum/num_batches_seen": curriculum_state.num_batches_seen.astype(jnp.float32),  # scalar
        }
    )
    return rng, agent_train_state, reward_normalization_stats, intrinsic_states, curriculum_state, diagnostics


def full_training(
    rng: jax.Array,
    agent_train_state: TrainState,
    reward_normalization_stats: RewardNormalizationStats,
    intrinsic_states: IntrinsicStates,
    curriculum_state: CurriculumState,
    env: Any,
    env_params: Any,
    intrinsic_modules: tuple[IntrinsicModule, ...],
    config: Any,
) -> dict[str, Any]:
    """Main training loop with fixed-alpha windows and intrinsic updates."""
    train_one_outer_iteration_jit = jax.jit(
        Partial(
            train_one_outer_iteration,
            env=env,
            env_params=env_params,
            intrinsic_modules=intrinsic_modules,
            config=config,
        )
    )

    metrics_per_batch = []
    for _ in range(config.num_batches_of_envs):
        rng, agent_train_state, reward_normalization_stats, intrinsic_states, curriculum_state, metrics = train_one_outer_iteration_jit(
            rng,
            agent_train_state,
            reward_normalization_stats,
            intrinsic_states,
            curriculum_state,
        )
        metrics_per_batch.append(metrics)

    metrics = {}
    if metrics_per_batch:
        metrics = jtu.tree_map(
            lambda *x: jnp.stack(x, axis=0),
            *metrics_per_batch,
        )

    return {
        "rng": rng,
        "agent_state": agent_train_state,
        "reward_normalization_stats": reward_normalization_stats,
        "intrinsic_states": intrinsic_states,
        "curriculum_state": curriculum_state,
        "metrics": metrics,
    }
