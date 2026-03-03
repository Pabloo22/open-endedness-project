"""Learning-progress estimation utilities from fixed-alpha rollout windows."""

import jax
import jax.numpy as jnp

from crew.main_algo.types import LpEstimationData


def _ols_slope(x: jax.Array, y: jax.Array, eps: float) -> jax.Array:
    """Ordinary least squares slope for points (x, y)."""
    x_mean = jnp.mean(x)
    y_mean = jnp.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    denominator = jnp.sum(x_centered**2)
    numerator = jnp.sum(x_centered * y_centered)
    return jnp.where(denominator > eps, numerator / denominator, jnp.array(0.0, dtype=y.dtype))


def _wls_slope(
    x: jax.Array,
    y: jax.Array,
    weights: jax.Array,
    valid_mask: jax.Array,
    eps: float,
) -> jax.Array:
    """Weighted least squares slope for weighted (x, y) points."""
    valid_weights = jnp.where(valid_mask, weights, jnp.array(0.0, dtype=weights.dtype))
    total_weight = jnp.sum(valid_weights)

    x_weighted_mean = jnp.where(
        total_weight > eps,
        jnp.sum(valid_weights * x) / total_weight,
        jnp.array(0.0, dtype=x.dtype),
    )
    y_weighted_mean = jnp.where(
        total_weight > eps,
        jnp.sum(valid_weights * y) / total_weight,
        jnp.array(0.0, dtype=y.dtype),
    )

    x_centered = x - x_weighted_mean
    y_centered = y - y_weighted_mean
    denominator = jnp.sum(valid_weights * (x_centered**2))
    numerator = jnp.sum(valid_weights * x_centered * y_centered)

    num_valid_points = jnp.sum(valid_mask.astype(jnp.int32))
    has_enough_points = num_valid_points >= 2
    has_denominator = denominator > eps
    return jnp.where(
        has_enough_points & has_denominator,
        numerator / denominator,
        jnp.array(0.0, dtype=y.dtype),
    )


def _estimate_lifetime_lp_for_stream(
    rewards_ut: jax.Array,
    gamma: jax.Array,
    eps: float,
) -> jax.Array:
    """Estimate raw signed LP slope for one lifetime-like reward stream (non-episodic).

    Shapes:
    - rewards_ut: [U, T]
    """
    # Mean reward per update is a proxy of steady-state reward rate under that policy.
    mean_reward_per_update = jnp.mean(rewards_ut, axis=1)  # [U]
    # Keep denominator numerically stable when gamma is very close to 1.
    discount_denom = jnp.maximum(
        jnp.array(1.0, dtype=rewards_ut.dtype) - gamma.astype(rewards_ut.dtype),
        jnp.array(eps, dtype=rewards_ut.dtype),
    )
    return_proxy_per_update = mean_reward_per_update / discount_denom  # [U]

    # x represents the sequence of policies over time.
    x = jnp.arange(rewards_ut.shape[0], dtype=rewards_ut.dtype)
    return _ols_slope(x=x, y=return_proxy_per_update, eps=eps)


def _estimate_episodic_lp_for_stream(
    rewards_ut: jax.Array,
    done_ut: jax.Array,
    eps: float,
) -> jax.Array:
    """Estimate raw signed LP slope for one episodic reward stream.
    Shapes:
    - rewards_ut, done_ut: [U, T]
    """
    num_updates = rewards_ut.shape[0]
    num_steps_per_update = rewards_ut.shape[1]
    total_steps = num_updates * num_steps_per_update

    rewards_flat = rewards_ut.reshape((total_steps,))  # [U*T]
    done_flat = done_ut.reshape((total_steps,))  # [U*T]
    update_indices_per_step = jnp.repeat(
        jnp.arange(num_updates, dtype=jnp.int32),
        num_steps_per_update,
    )  # [U*T]
    update_indices_float = jnp.arange(num_updates, dtype=rewards_ut.dtype)  # [U]

    def scan_step(carry, step_inputs):
        """Track one ongoing episode and emit exactly one (x,y) point when done=True."""
        episode_return, steps_per_update = carry
        reward_t, done_t, update_idx_t = step_inputs

        # Extend the current episode with this step.
        updated_episode_return = episode_return + reward_t
        updated_steps_per_update = steps_per_update.at[update_idx_t].add(1)

        # Fraction of the episode executed under each update index.
        total_episode_steps = jnp.sum(updated_steps_per_update)
        total_episode_steps = jnp.maximum(total_episode_steps, jnp.array(1, dtype=jnp.int32))
        fractions_per_update = updated_steps_per_update.astype(rewards_ut.dtype) / total_episode_steps.astype(
            rewards_ut.dtype
        )
        # Effective policy index for this episode (fractional for cross-update episodes).
        episode_x = jnp.sum(fractions_per_update * update_indices_float)

        # Intra-update episodes are those that touched only one update index (were collected by only one policy - no updates in the middle).
        num_touched_updates = jnp.sum((updated_steps_per_update > 0).astype(jnp.int32))
        episode_is_intra = num_touched_updates == 1
        episode_intra_update_idx = jnp.argmax(updated_steps_per_update).astype(jnp.int32)

        # Emit episode stats only on terminal steps.
        emitted_valid = done_t
        emitted_x = jnp.where(emitted_valid, episode_x, jnp.array(0.0, dtype=rewards_ut.dtype))
        emitted_y = jnp.where(emitted_valid, updated_episode_return, jnp.array(0.0, dtype=rewards_ut.dtype))
        emitted_is_intra = emitted_valid & episode_is_intra
        emitted_intra_update_idx = jnp.where(
            emitted_valid,
            episode_intra_update_idx,
            jnp.array(-1, dtype=jnp.int32),
        )

        # Reset at episode termination.
        next_episode_return = jnp.where(
            done_t,
            jnp.array(0.0, dtype=rewards_ut.dtype),
            updated_episode_return,
        )
        next_steps_per_update = jnp.where(
            done_t,
            jnp.zeros_like(updated_steps_per_update),
            updated_steps_per_update,
        )

        next_carry = (next_episode_return, next_steps_per_update)
        step_outputs = (
            emitted_valid,
            emitted_x,
            emitted_y,
            emitted_is_intra,
            emitted_intra_update_idx,
        )
        return next_carry, step_outputs

    init_carry = (
        jnp.array(0.0, dtype=rewards_ut.dtype),
        jnp.zeros((num_updates,), dtype=jnp.int32),
    )
    _, step_outputs = jax.lax.scan(
        scan_step,
        init_carry,
        (rewards_flat, done_flat, update_indices_per_step),
    )
    (
        episode_valid_per_step,
        episode_x_per_step,
        episode_return_per_step,
        episode_is_intra_per_step,
        episode_intra_update_idx_per_step,
    ) = step_outputs

    # Aggregate points from episodes that were collected by the same policy
    # Episodes collected by multiple policies (policy updated during collection) are kept as individual points
    intra_episode_mask = episode_valid_per_step & episode_is_intra_per_step
    clipped_intra_idx = jnp.clip(episode_intra_update_idx_per_step, 0, num_updates - 1)
    intra_one_hot = jax.nn.one_hot(
        clipped_intra_idx,
        num_updates,
        dtype=rewards_ut.dtype,
    )  # [U*T, U]
    intra_episode_mask_float = intra_episode_mask.astype(rewards_ut.dtype)

    intra_counts = jnp.sum(intra_one_hot * intra_episode_mask_float[:, None], axis=0)  # [U]
    intra_returns_sum = jnp.sum(
        intra_one_hot * intra_episode_mask_float[:, None] * episode_return_per_step[:, None],
        axis=0,
    )  # [U]
    intra_returns_mean = intra_returns_sum / jnp.maximum(intra_counts, jnp.array(1.0, dtype=rewards_ut.dtype))
    intra_weights = jnp.sqrt(intra_counts)
    intra_valid = intra_counts > 0

    multi_episode_mask = episode_valid_per_step & jnp.logical_not(episode_is_intra_per_step)
    multi_x = episode_x_per_step
    multi_y = episode_return_per_step
    multi_weights = jnp.ones_like(multi_y)

    # Build the final weighted regression dataset.
    x_points = jnp.concatenate((update_indices_float, multi_x), axis=0)
    y_points = jnp.concatenate((intra_returns_mean, multi_y), axis=0)
    w_points = jnp.concatenate((intra_weights, multi_weights), axis=0)
    valid_points = jnp.concatenate((intra_valid, multi_episode_mask), axis=0)

    return _wls_slope(
        x=x_points,
        y=y_points,
        weights=w_points,
        valid_mask=valid_points,
        eps=eps,
    )


def _estimate_lp_single_env_single_reward(
    rewards_ut: jax.Array,
    done_ut: jax.Array,
    is_episodic: jax.Array,
    gamma: jax.Array,
    eps: float,
) -> jax.Array:
    """Estimate raw signed LP slope for one env and one reward function."""
    return jax.lax.cond(
        is_episodic,
        lambda _: _estimate_episodic_lp_for_stream(rewards_ut=rewards_ut, done_ut=done_ut, eps=eps),
        lambda _: _estimate_lifetime_lp_for_stream(rewards_ut=rewards_ut, gamma=gamma, eps=eps),
        operand=None,
    )


def estimate_lp_per_reward_function(
    lp_estimation_data: LpEstimationData,
    is_episodic_per_reward_function: jax.Array,
    gamma_per_reward_function: jax.Array,
    eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    """Estimate raw signed LP slope for each env/reward pair.

    Key Input Shapes:
    - lp_estimation_data.raw_rewards, done_masks: [U, T, B, R]
    - is_episodic_per_reward_function: [R]
    - gamma_per_reward_function: [R]
    - returns:
      - lp_per_reward_function: [B, R] (raw slope, unnormalized)
      - insufficient_episodes_env_mask: [B], int32 in {0,1}
    """
    raw_rewards = lp_estimation_data.raw_rewards
    done_masks = lp_estimation_data.done_masks

    # Reorder to [B, R, U, T].
    rewards_brut = jnp.transpose(raw_rewards, (2, 3, 0, 1))
    done_masks_brut = jnp.transpose(done_masks, (2, 3, 0, 1))

    # Envs with fewer than two completed episodes are not valid for score targets.
    completed_episodes_per_env = jnp.sum(done_masks[..., 0].astype(jnp.int32), axis=(0, 1))  # [B]
    insufficient_episodes_env_mask = (completed_episodes_per_env < 2).astype(jnp.int32)  # [B]

    def estimate_for_single_env(
        rewards_rut: jax.Array,
        done_masks_rut: jax.Array,
    ) -> jax.Array:
        return jax.vmap(
            _estimate_lp_single_env_single_reward,
            in_axes=(0, 0, 0, 0, None),
        )(
            rewards_rut,
            done_masks_rut,
            is_episodic_per_reward_function,
            gamma_per_reward_function,
            eps,
        )

    lp_per_reward_function = jax.vmap(
        estimate_for_single_env,
        in_axes=(0, 0),
    )(rewards_brut, done_masks_brut)
    return lp_per_reward_function, insufficient_episodes_env_mask
