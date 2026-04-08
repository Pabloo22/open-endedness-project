"""Alpha-score replay buffer helpers for curriculum training data."""

from typing import Any

import jax
import jax.numpy as jnp

from curemix.main_algo.types import AlphaScoreReplayBuffer


def init_alpha_score_replay_buffer(config: Any) -> AlphaScoreReplayBuffer:
    """Initialize a fixed-size ring buffer for (alpha, score, validity) datapoints."""
    capacity = int(config.curriculum.replay_buffer_num_batches * config.num_envs_per_batch)
    return AlphaScoreReplayBuffer(
        alpha=jnp.zeros((capacity, config.num_reward_functions), dtype=jnp.float32),
        score=jnp.zeros((capacity,), dtype=jnp.float32),
        is_valid=jnp.zeros((capacity,), dtype=jnp.bool_),
        write_index=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
        capacity=capacity,
        batch_size=config.num_envs_per_batch,
    )


def add_alpha_score_batch(
    alpha_score_replay_buffer: AlphaScoreReplayBuffer,
    alpha_batch: jax.Array,
    score_batch: jax.Array,
    is_valid_batch: jax.Array,
) -> AlphaScoreReplayBuffer:
    """Insert one full outer-loop batch into the ring buffer.

    Shapes:
    - alpha_batch: [B, R]
    - score_batch, is_valid_batch: [B]
    - B must match alpha_score_replay_buffer.batch_size
    """
    capacity = alpha_score_replay_buffer.capacity
    insert_indices = (
        alpha_score_replay_buffer.write_index + jnp.arange(alpha_score_replay_buffer.batch_size, dtype=jnp.int32)
    ) % capacity

    next_alpha = alpha_score_replay_buffer.alpha.at[insert_indices].set(
        alpha_batch.astype(alpha_score_replay_buffer.alpha.dtype)
    )
    next_score = alpha_score_replay_buffer.score.at[insert_indices].set(
        score_batch.astype(alpha_score_replay_buffer.score.dtype)
    )
    next_valid = alpha_score_replay_buffer.is_valid.at[insert_indices].set(is_valid_batch.astype(jnp.bool_))

    next_write_index = (alpha_score_replay_buffer.write_index + alpha_score_replay_buffer.batch_size) % capacity
    next_size = jnp.minimum(alpha_score_replay_buffer.size + alpha_score_replay_buffer.batch_size, capacity)

    return alpha_score_replay_buffer.replace(
        alpha=next_alpha,
        score=next_score,
        is_valid=next_valid,
        write_index=next_write_index,
        size=next_size,
    )
