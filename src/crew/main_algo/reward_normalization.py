import jax
import jax.numpy as jnp

from crew.main_algo.types import RewardNormalizationStats


def init_reward_normalization_stats(num_envs: int, num_reward_functions: int) -> RewardNormalizationStats:
    """Initialize running reward-normalization statistics.

    Key output shapes:
    - running_forward_return: [B, R]
    - mean, M2, var: [R]
    """
    return RewardNormalizationStats(
        running_forward_return=jnp.zeros((num_envs, num_reward_functions), dtype=jnp.float32),
        count=jnp.array(0.0, dtype=jnp.float32),
        mean=jnp.zeros((num_reward_functions,), dtype=jnp.float32),
        M2=jnp.zeros((num_reward_functions,), dtype=jnp.float32),
        var=jnp.ones((num_reward_functions,), dtype=jnp.float32),
    )


def compute_forward_returns(
    running_forward_return: jax.Array,
    rewards: jax.Array,
    gamma: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute forward discounted returns used for reward normalization.

    Key input shapes:
    - running_forward_return: [B, R]
    - rewards: [T, B, R]
    - gamma: [R]

    Key output shapes:
    - new_running_forward_return: [B, R]
    - forward_returns: [T, B, R]
    """

    def scan_step(running_returns, rewards_t):
        new_running_returns = rewards_t + gamma[None, :] * running_returns
        return new_running_returns, new_running_returns

    new_running_forward_return, forward_returns = jax.lax.scan(scan_step, running_forward_return, rewards, reverse=False)
    return new_running_forward_return, forward_returns


def update_reward_normalization_stats(old_stats: RewardNormalizationStats, forward_returns: jax.Array) -> RewardNormalizationStats:
    """Update running mean/variance of forward returns (for each reward function) using Chan's batch version of Welford's algorithm.

    Key input shapes:
    - old_stats.running_forward_return: [B, R]
    - old_stats.mean, old_stats.M2, old_stats.var: [R]
    - forward_returns: [T, B, R]

    Key output shapes:
    - new_stats.mean, new_stats.M2, new_stats.var: [R]
    """
    flattened_returns = forward_returns.reshape((-1, forward_returns.shape[-1]))  # [T*B, R]
    batch_count = jnp.array(flattened_returns.shape[0], dtype=old_stats.count.dtype)
    batch_mean = jnp.mean(flattened_returns, axis=0)
    batch_M2 = jnp.sum((flattened_returns - batch_mean[None, :]) ** 2, axis=0)

    new_count = old_stats.count + batch_count
    delta = batch_mean - old_stats.mean
    new_mean = old_stats.mean + delta * batch_count / new_count
    new_M2 = old_stats.M2 + batch_M2 + ((delta**2) * (old_stats.count * batch_count / new_count))
    new_var = new_M2 / new_count

    return old_stats.replace(count=new_count, mean=new_mean, M2=new_M2, var=new_var)


def normalize_rewards(
    rewards: jax.Array,
    stats: RewardNormalizationStats,
    eps: float,
    clip: float | None,
) -> jax.Array:
    """Normalize rewards using running per-reward-function return std.

    Key input shapes:
    - rewards: [T, B, R]
    - stats.var: [R]

    Key output shapes:
    - normalized_rewards: [T, B, R]
    """
    denom = jnp.sqrt(stats.var + eps)[None, None, :]
    normalized_rewards = rewards / denom
    if clip is not None:
        normalized_rewards = jnp.clip(normalized_rewards, -clip, clip)
    return normalized_rewards
