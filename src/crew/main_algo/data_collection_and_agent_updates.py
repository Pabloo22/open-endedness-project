from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from crew.main_algo.actor_critic import ActorCriticTransformer
from crew.main_algo.advantage_computation import (
    compute_gae,
    compute_weighted_advantages,
    normalize_advantages,
)
from crew.main_algo.intrinsic_modules.api import IntrinsicModule
from crew.main_algo.ppo import update_agent
from crew.main_algo.reward_normalization import (
    compute_forward_returns,
    normalize_rewards,
    update_reward_normalization_stats,
)
from crew.main_algo.types import (
    IntrinsicModulesUpdateData,
    IntrinsicStates,
    LpEstimationData,
    RewardNormalizationStats,
    RunnerStateTransformer,
    TransitionDataBase,
    TransitionDataTransformer,
)


def advance_memories_mask(
    memories_mask: jax.Array,
    memories_mask_idx: jax.Array,
    prev_done: jax.Array,
    config: Any,
) -> tuple[jax.Array, jax.Array]:
    """Advance/reset transformer mask to match the next observation step."""
    next_memories_mask_idx = jnp.where(
        prev_done,
        config.past_context_length,
        jnp.clip(memories_mask_idx - 1, 0, config.past_context_length),
    )
    next_memories_mask = jnp.where(
        prev_done[:, None, None, None],
        jnp.zeros(
            (
                config.num_envs_per_batch,
                config.num_attn_heads,
                1,
                config.past_context_length + 1,
            ),
            dtype=jnp.bool_,
        ),
        memories_mask,
    )
    next_memories_mask_idx_one_hot = jax.nn.one_hot(
        next_memories_mask_idx,
        config.past_context_length + 1,
        dtype=jnp.bool_,
    )
    next_memories_mask_idx_one_hot = jnp.repeat(
        next_memories_mask_idx_one_hot[:, None, None, :],
        config.num_attn_heads,
        axis=1,
    )
    next_memories_mask = jnp.logical_or(next_memories_mask, next_memories_mask_idx_one_hot)
    return next_memories_mask, next_memories_mask_idx


def compute_intrinsic_rewards_and_done_masks(
    rng: jax.Array,
    intrinsic_modules: tuple[IntrinsicModule, ...],
    intrinsic_states: IntrinsicStates,
    transitions: TransitionDataBase,
    config: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute reward and done tensors for all reward functions.

    Key input shapes:
    - transitions.reward, transitions.done: [T, B]
    - each intrinsic module reward and done mask: [T, B]

    Key output shapes:
    - rewards: [T, B, R] with index 0 extrinsic and 1..R-1 intrinsic modules
    - done: [T, B, R] aligned with rewards
    """
    next_rng, *intrinsic_rngs = jax.random.split(rng, len(intrinsic_modules) + 1)

    intrinsic_rewards = tuple(
        module.compute_rewards(module_rng, module_state, transitions, config)
        for module, module_state, module_rng in zip(
            intrinsic_modules,
            intrinsic_states,
            intrinsic_rngs,
            strict=True,
        )
    )
    intrinsic_done_masks = tuple(module.done_mask(transitions.done, config) for module in intrinsic_modules)

    rewards = jnp.stack((transitions.reward, *intrinsic_rewards), axis=-1)
    done = jnp.stack((transitions.done, *intrinsic_done_masks), axis=-1)
    return next_rng, rewards, done


def normalize_rewards_and_update_normalization_stats(
    rewards: jax.Array,
    done: jax.Array,
    reward_normalization_stats: RewardNormalizationStats,
    config: Any,
) -> tuple[RewardNormalizationStats, jax.Array, dict[str, jax.Array]]:
    """Normalize rewards and update running normalization statistics.

    Key input shapes:
    - rewards: [T, B, R]
    - done: [T, B, R]
    - reward_normalization_stats.running_forward_return: [B, R]
    - config.gamma_per_reward_function: [R]

    Key output shapes:
    - updated_reward_normalization_stats: pytree with [B, R] and [R] stats
    - rewards_normalized: [T, B, R]
    """
    new_running_forward_return, new_previous_done, forward_returns = compute_forward_returns(
        reward_normalization_stats.running_forward_return,
        reward_normalization_stats.previous_done,
        rewards,
        done,
        config.gamma_per_reward_function,
    )
    reward_normalization_stats = reward_normalization_stats.replace(
        running_forward_return=new_running_forward_return,
        previous_done=new_previous_done,
    )
    reward_normalization_stats = update_reward_normalization_stats(reward_normalization_stats, forward_returns)

    rewards_normalized = normalize_rewards(
        rewards=rewards,
        stats=reward_normalization_stats,
        eps=config.reward_norm_eps,
        clip=config.reward_norm_clip,
    )
    metrics: dict[str, jax.Array] = {}
    return reward_normalization_stats, rewards_normalized, metrics


def compute_value_targets_and_weighted_advantages(
    rewards_normalized: jax.Array,
    done: jax.Array,
    values: jax.Array,
    last_values: jax.Array,
    alpha_batch: jax.Array,
    config: Any,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Compute raw/normalized advantages, value targets, and weighted advantage.

    Key input shapes:
    - rewards_normalized, done, values: [T, B, R]
    - last_values, alpha_batch: [B, R]
    - config.gamma_per_reward_function, config.gae_lambda_per_reward_function: [R]

    Key output shapes:
    - advantages_raw, advantages_normalized, value_targets: [T, B, R]
    - weighted_advantages: [T, B]
    """
    advantages_raw, value_targets = compute_gae(
        rewards=rewards_normalized,
        values=values,
        last_values=last_values,
        done=done,
        gamma=config.gamma_per_reward_function,
        gae_lambda=config.gae_lambda_per_reward_function,
    )
    advantages_normalized = normalize_advantages(
        advantages=advantages_raw,
        eps=config.adv_norm_eps,
    )
    weighted_advantages = compute_weighted_advantages(
        alpha_batch=alpha_batch,
        normalized_advantages=advantages_normalized,
    )
    metrics = {
        "preproc/adv_raw_mean": jnp.mean(advantages_raw, axis=(0, 1)),
        "preproc/adv_norm_mean": jnp.mean(advantages_normalized, axis=(0, 1)),
        "preproc/adv_norm_std": jnp.std(advantages_normalized, axis=(0, 1)),
        "preproc/weighted_adv_mean": jnp.mean(weighted_advantages),
        "preproc/weighted_adv_std": jnp.std(weighted_advantages),
    }
    return value_targets, weighted_advantages, metrics


def step_envs(
    runner_state: RunnerStateTransformer,
    current_update_step_num: jax.Array,
    env: Any,
    env_params: Any,
    alpha_batch: jax.Array,
    config: Any,
) -> tuple[RunnerStateTransformer, tuple[TransitionDataTransformer, jax.Array]]:
    """Step all environments once and record rollout data."""
    memories_mask, memories_mask_idx = advance_memories_mask(
        memories_mask=runner_state.memories_mask,
        memories_mask_idx=runner_state.memories_mask_idx,
        prev_done=runner_state.prev_done,
        config=config,
    )

    # Policy/value evaluation for current observations.
    rng, action_rng = jax.random.split(runner_state.rng)
    pi, values, memories_out = runner_state.agent_train_state.apply_fn(
        runner_state.agent_train_state.params,
        runner_state.memories,
        runner_state.prev_obs[:, None, :],
        memories_mask,
        alpha_batch,
        method=ActorCriticTransformer.model_forward_eval,
    )
    action = pi.sample(seed=action_rng)
    log_prob = pi.log_prob(action)

    # Update rolling memory cache with the new transformer outputs.
    memories = jnp.roll(runner_state.memories, -1, axis=1).at[:, -1].set(memories_out)

    # Step environments
    rng, step_rng_base = jax.random.split(rng)
    step_rngs = jax.random.split(step_rng_base, num=config.num_envs_per_batch)
    next_obs, env_state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        step_rngs,
        runner_state.env_state,
        action,
        env_params,
    )

    # Indices used in PPO.
    memories_indices = jnp.arange(0, config.past_context_length)[None, :] + current_update_step_num * jnp.ones(
        (config.num_envs_per_batch, 1),
        dtype=jnp.int32,
    )

    transition = TransitionDataTransformer(
        obs=runner_state.prev_obs,
        next_obs=next_obs,
        action=action,
        done=done,
        reward=reward,
        value=values,
        log_prob=log_prob,
        memories_mask=memories_mask.squeeze(axis=2),
        memories_indices=memories_indices,
    )

    runner_state = runner_state.replace(
        rng=rng,
        env_state=env_state,
        prev_obs=next_obs,
        prev_done=done,
        memories=memories,
        memories_mask=memories_mask,
        memories_mask_idx=memories_mask_idx,
    )
    return runner_state, (transition, memories_out)


def collect_data(
    runner_state: RunnerStateTransformer,
    num_steps: int,
    alpha_batch: jax.Array,
    env: Any,
    env_params: Any,
    config: Any,
) -> tuple[RunnerStateTransformer, TransitionDataTransformer, jax.Array]:
    """Collect a rollout window of length num_steps."""
    runner_state, (transitions, memories_batch) = jax.lax.scan(
        Partial(
            step_envs,
            env=env,
            env_params=env_params,
            alpha_batch=alpha_batch,
            config=config,
        ),
        runner_state,
        jnp.arange(num_steps, dtype=jnp.int32),
    )
    return runner_state, transitions, memories_batch


def collect_data_and_update_agent(
    runner_state: RunnerStateTransformer,
    _unused: Any,
    env: Any,
    env_params: Any,
    alpha_batch: jax.Array,
    intrinsic_states: IntrinsicStates,
    intrinsic_modules: tuple[IntrinsicModule, ...],
    config: Any,
) -> tuple[
    RunnerStateTransformer,
    tuple[dict[str, jax.Array], IntrinsicModulesUpdateData, LpEstimationData],
]:
    """Collect rollout data, preprocess rewards/advantages, and run PPO updates."""
    memories_previous = runner_state.memories

    runner_state, transitions, memories_batch = collect_data(
        runner_state=runner_state,
        num_steps=config.num_steps_per_update,
        alpha_batch=alpha_batch,
        env=env,
        env_params=env_params,
        config=config,
    )

    rng, rewards, done = compute_intrinsic_rewards_and_done_masks(
        rng=runner_state.rng,
        intrinsic_modules=intrinsic_modules,
        intrinsic_states=intrinsic_states,
        transitions=transitions,
        config=config,
    )

    reward_normalization_stats, rewards_normalized, reward_metrics = normalize_rewards_and_update_normalization_stats(
        rewards=rewards,
        done=done,
        reward_normalization_stats=runner_state.reward_normalization_stats,
        config=config,
    )

    bootstrap_memories_mask, _ = advance_memories_mask(
        memories_mask=runner_state.memories_mask,
        memories_mask_idx=runner_state.memories_mask_idx,
        prev_done=runner_state.prev_done,
        config=config,
    )

    _, last_values, _ = runner_state.agent_train_state.apply_fn(
        runner_state.agent_train_state.params,
        runner_state.memories,
        runner_state.prev_obs[:, None, :],
        bootstrap_memories_mask,
        alpha_batch,
        method=ActorCriticTransformer.model_forward_eval,
    )

    value_targets, weighted_advantages, advantage_metrics = compute_value_targets_and_weighted_advantages(
        rewards_normalized=rewards_normalized,
        done=done,
        values=transitions.value,
        last_values=last_values,
        alpha_batch=alpha_batch,
        config=config,
    )

    memories_batch = jnp.concatenate(
        [jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0
    )  # [past_context + num_steps_per_update, num_envs, num_tranformer_layers, hidden]

    rng, agent_train_state, ppo_metrics = update_agent(
        rng=rng,
        agent_train_state=runner_state.agent_train_state,
        transitions=transitions,
        memories_batch=memories_batch,
        alpha_batch=alpha_batch,
        weighted_advantages=weighted_advantages,
        value_targets=value_targets,
        config=config,
    )

    # prepare outputs
    metrics = reward_metrics | advantage_metrics | ppo_metrics

    intrinsic_modules_update_data = IntrinsicModulesUpdateData(
        obs=transitions.obs,
        next_obs=transitions.next_obs,
        action=transitions.action,
        done=transitions.done,
    )

    lp_estimation_data = LpEstimationData(
        raw_rewards=rewards,
        done_masks=done,
    )

    updated_runner_state = runner_state.replace(
        rng=rng,
        agent_train_state=agent_train_state,
        reward_normalization_stats=reward_normalization_stats,
    )

    return (
        updated_runner_state,
        (
            metrics,
            intrinsic_modules_update_data,
            lp_estimation_data,
        ),
    )
