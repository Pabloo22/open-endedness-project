"""Learning-progress normalization statistics.

Tracks per-reward-function return-scale estimates for LP normalization.
"""

import jax
import jax.numpy as jnp

from crew.main_algo.types import LpEstimationData, LpNormalizationStats


def init_lp_normalization_stats(num_reward_functions: int) -> LpNormalizationStats:
    """Initialize LP normalization stats.

    Shapes:
    - mean, second_moment, var, initialized: [R]
    """
    return LpNormalizationStats(
        mean=jnp.zeros((num_reward_functions,), dtype=jnp.float32),
        second_moment=jnp.ones((num_reward_functions,), dtype=jnp.float32),
        var=jnp.ones((num_reward_functions,), dtype=jnp.float32),
        initialized=jnp.zeros((num_reward_functions,), dtype=jnp.bool_),
    )


def _compute_episodic_batch_moments(
    rewards_sb: jax.Array,
    done_sb: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute batch moments for episodic streams using completed episode returns.

    Shapes:
    - rewards_sb, done_sb: [S, B]
    - returns:
      - batch_mean: scalar
      - batch_second_moment: scalar
      - is_valid: scalar bool
    """

    def scan_step(carry, step_inputs):
        # Shapes for all carry terms: [B]
        running_episode_return, episode_count, episode_return_sum, episode_return_sq_sum = carry
        reward_t, done_t = step_inputs

        updated_running_return = running_episode_return + reward_t
        episode_count = episode_count + done_t.astype(jnp.int32)
        episode_return_sum = episode_return_sum + jnp.where(done_t, updated_running_return, jnp.array(0.0, reward_t.dtype))
        episode_return_sq_sum = episode_return_sq_sum + jnp.where(
            done_t,
            updated_running_return**2,
            jnp.array(0.0, reward_t.dtype),
        )
        next_running_return = jnp.where(done_t, jnp.array(0.0, reward_t.dtype), updated_running_return)
        return (next_running_return, episode_count, episode_return_sum, episode_return_sq_sum), None

    batch_size = rewards_sb.shape[1]
    init_carry = (
        jnp.zeros((batch_size,), dtype=rewards_sb.dtype),
        jnp.zeros((batch_size,), dtype=jnp.int32),
        jnp.zeros((batch_size,), dtype=rewards_sb.dtype),
        jnp.zeros((batch_size,), dtype=rewards_sb.dtype),
    )
    (running_return, episode_count, episode_return_sum, episode_return_sq_sum), _ = jax.lax.scan(
        scan_step,
        init_carry,
        (rewards_sb, done_sb),
    )
    del running_return

    has_episodes_per_env = episode_count > 0  # [B]
    active_env_count = jnp.sum(has_episodes_per_env.astype(jnp.int32))
    denom_per_env = jnp.maximum(episode_count.astype(rewards_sb.dtype), jnp.array(1.0, dtype=rewards_sb.dtype))
    env_mean = episode_return_sum / denom_per_env  # [B]
    env_second_moment = episode_return_sq_sum / denom_per_env  # [B]

    active_env_count_float = jnp.maximum(active_env_count.astype(rewards_sb.dtype), jnp.array(1.0, dtype=rewards_sb.dtype))
    active_env_mask = has_episodes_per_env.astype(rewards_sb.dtype)
    batch_mean = jnp.sum(env_mean * active_env_mask) / active_env_count_float
    batch_second_moment = jnp.sum(env_second_moment * active_env_mask) / active_env_count_float
    is_valid = active_env_count > 0
    return batch_mean, batch_second_moment, is_valid


def _compute_continuous_batch_moments(
    rewards_sb: jax.Array,
    gamma: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute batch moments for non-episodic streams from forward returns.

    Shapes:
    - rewards_sb: [S, B]
    - gamma: scalar
    - returns:
      - batch_mean: scalar
      - batch_second_moment: scalar
      - is_valid: scalar bool
    """
    mean_reward_per_env = jnp.mean(rewards_sb, axis=0)  # [B]
    discount_denom = jnp.maximum(
        jnp.array(1.0, dtype=rewards_sb.dtype) - gamma.astype(rewards_sb.dtype),
        jnp.array(eps, dtype=rewards_sb.dtype),
    )
    init_forward_return = mean_reward_per_env / discount_denom  # [B]

    def scan_step(running_forward_return, reward_t):
        new_running_forward_return = reward_t + gamma.astype(reward_t.dtype) * running_forward_return
        return new_running_forward_return, new_running_forward_return

    _, forward_returns = jax.lax.scan(scan_step, init_forward_return, rewards_sb)
    batch_mean = jnp.mean(forward_returns)
    batch_second_moment = jnp.mean(forward_returns**2)
    is_valid = jnp.array(True, dtype=jnp.bool_)
    return batch_mean, batch_second_moment, is_valid


def compute_lp_batch_moments(
    lp_estimation_data: LpEstimationData,
    is_episodic_per_reward_function: jax.Array,
    gamma_per_reward_function: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute per-reward batch moments used to update LP normalization stats.

    Shapes:
    - lp_estimation_data.raw_rewards, done_masks: [U, T, B, R]
    - is_episodic_per_reward_function, gamma_per_reward_function: [R]
    - returns:
      - batch_mean_per_reward: [R]
      - batch_second_moment_per_reward: [R]
      - valid_mask_per_reward: [R], bool
    """
    raw_rewards = lp_estimation_data.raw_rewards
    done_masks = lp_estimation_data.done_masks
    num_reward_functions = raw_rewards.shape[-1]
    flattened_steps = raw_rewards.shape[0] * raw_rewards.shape[1]
    batch_size = raw_rewards.shape[2]

    rewards_rsb = jnp.reshape(
        jnp.transpose(raw_rewards, (3, 0, 1, 2)),
        (num_reward_functions, flattened_steps, batch_size),
    )  # [R, S, B]
    done_rsb = jnp.reshape(
        jnp.transpose(done_masks, (3, 0, 1, 2)),
        (num_reward_functions, flattened_steps, batch_size),
    )  # [R, S, B]

    def compute_for_reward(rewards_sb, done_sb, is_episodic, gamma):
        return jax.lax.cond(
            is_episodic,
            lambda _: _compute_episodic_batch_moments(rewards_sb=rewards_sb, done_sb=done_sb),
            lambda _: _compute_continuous_batch_moments(rewards_sb=rewards_sb, gamma=gamma, eps=eps),
            operand=None,
        )

    return jax.vmap(compute_for_reward, in_axes=(0, 0, 0, 0))(
        rewards_rsb,
        done_rsb,
        is_episodic_per_reward_function,
        gamma_per_reward_function,
    )


def update_lp_normalization_stats(
    old_stats: LpNormalizationStats,
    batch_mean_per_reward: jax.Array,
    batch_second_moment_per_reward: jax.Array,
    valid_mask_per_reward: jax.Array,
    ema_beta: float,
) -> LpNormalizationStats:
    """Update LP normalization statistics with per-reward EMA and first-batch bootstrap.

    Shapes:
    - old_stats.mean, second_moment, var, initialized: [R]
    - batch_mean_per_reward, batch_second_moment_per_reward, valid_mask_per_reward: [R]
    """
    beta = jnp.array(ema_beta, dtype=old_stats.mean.dtype)
    should_bootstrap = valid_mask_per_reward & jnp.logical_not(old_stats.initialized)
    should_ema_update = valid_mask_per_reward & old_stats.initialized

    mean_after_ema = (jnp.array(1.0, dtype=old_stats.mean.dtype) - beta) * old_stats.mean + beta * batch_mean_per_reward
    second_after_ema = (jnp.array(1.0, dtype=old_stats.mean.dtype) - beta) * old_stats.second_moment + beta * batch_second_moment_per_reward

    new_mean = jnp.where(should_ema_update, mean_after_ema, old_stats.mean)
    new_mean = jnp.where(should_bootstrap, batch_mean_per_reward, new_mean)

    new_second_moment = jnp.where(should_ema_update, second_after_ema, old_stats.second_moment)
    new_second_moment = jnp.where(should_bootstrap, batch_second_moment_per_reward, new_second_moment)

    new_initialized = old_stats.initialized | valid_mask_per_reward
    new_var = jnp.maximum(new_second_moment - new_mean**2, jnp.array(0.0, dtype=old_stats.mean.dtype))
    return old_stats.replace(
        mean=new_mean,
        second_moment=new_second_moment,
        var=new_var,
        initialized=new_initialized,
    )


def update_lp_normalization_stats_from_data(
    old_stats: LpNormalizationStats,
    lp_estimation_data: LpEstimationData,
    is_episodic_per_reward_function: jax.Array,
    gamma_per_reward_function: jax.Array,
    ema_beta: float,
    eps: float,
) -> LpNormalizationStats:
    """Compute batch moments from LP data and update LP normalization stats."""
    batch_mean, batch_second_moment, valid_mask = compute_lp_batch_moments(
        lp_estimation_data=lp_estimation_data,
        is_episodic_per_reward_function=is_episodic_per_reward_function,
        gamma_per_reward_function=gamma_per_reward_function,
        eps=eps,
    )
    return update_lp_normalization_stats(
        old_stats=old_stats,
        batch_mean_per_reward=batch_mean,
        batch_second_moment_per_reward=batch_second_moment,
        valid_mask_per_reward=valid_mask,
        ema_beta=ema_beta,
    )

