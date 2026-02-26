import jax
import jax.numpy as jnp


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    last_values: jax.Array,
    done: jax.Array,
    gamma: jax.Array,
    gae_lambda: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute GAE and value targets per reward function in parallel.

    Key input shapes:
    - rewards, values, done: [T, B, R]
    - last_values: [B, R]
    - gamma, gae_lambda: [R]

    Key output shapes:
    - advantages: [T, B, R]
    - value_targets: [T, B, R]
    """

    def scan_step(carry, transition_t):
        gae, next_values = carry
        rewards_t, values_t, done_t = transition_t
        not_done = 1.0 - done_t.astype(values_t.dtype)
        delta = rewards_t + gamma[None, :] * next_values * not_done - values_t
        gae = delta + gamma[None, :] * gae_lambda[None, :] * not_done * gae
        return (gae, values_t), gae

    _, advantages = jax.lax.scan(
        scan_step,
        (jnp.zeros_like(last_values), last_values),
        (rewards, values, done),
        reverse=True,
    )
    value_targets = advantages + values
    return advantages, value_targets


def normalize_advantages(advantages: jax.Array, eps: float) -> jax.Array:
    """Normalize each reward function advantage over the full [T, B] batch.

    Key input shapes:
    - advantages: [T, B, R]

    Key output shapes:
    - normalized_advantages: [T, B, R]
    """
    mean = jnp.mean(advantages, axis=(0, 1))
    std = jnp.std(advantages, axis=(0, 1))
    return (advantages - mean[None, None, :]) / (std[None, None, :] + eps)


def compute_weighted_advantages(alpha_batch: jax.Array, normalized_advantages: jax.Array) -> jax.Array:
    """Merge advantages from each reward function into a single alpha-weighted advantage.

    Key input shapes:
    - alpha_batch: [B, R]
    - normalized_advantages: [T, B, R]

    Key output shapes:
    - weighted_advantage: [T, B]
    """
    return jnp.einsum("tbr,br->tb", normalized_advantages, alpha_batch)
