from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState

# Protocol-facing state alias. Each intrinsic module owns its concrete state pytree.
IntrinsicModuleState: TypeAlias = Any

# Standard metrics mapping returned by intrinsic module updates.
IntrinsicUpdateMetrics: TypeAlias = dict[str, jax.Array]

# Per-run intrinsic states aligned with the statically selected module tuple.
IntrinsicStates: TypeAlias = tuple[IntrinsicModuleState, ...]


class RewardNormalizationStats(struct.PyTreeNode):
    """Running return-normalization statistics kept per reward function.

    Shapes:
    - running_forward_return: [B, R] where R = num_reward_functions
    - previous_done: [B, R] boolean done mask from the previous step
    - mean, M2, var: [R]
    - count: scalar
    """

    running_forward_return: jax.Array
    previous_done: jax.Array
    count: jax.Array
    mean: jax.Array
    M2: jax.Array
    var: jax.Array


class TransitionDataBase(struct.PyTreeNode):
    # when collected with 'collect_data' function:
    # obs, next_obs: [T, B, *obs_shape]
    # action, done, reward, log_prob: [T, B]
    # value: [T, B, R]
    obs: jax.Array
    next_obs: jax.Array
    action: jax.Array
    done: jax.Array
    reward: jax.Array
    value: jax.Array
    log_prob: jax.Array


class TransitionDataTransformer(TransitionDataBase):
    memories_mask: jnp.ndarray
    memories_indices: jnp.ndarray


class IntrinsicModulesUpdateData(struct.PyTreeNode):
    """Subset of rollout data used to update intrinsic reward modules."""

    obs: jax.Array
    next_obs: jax.Array
    action: jax.Array
    done: jax.Array


class LpEstimationData(struct.PyTreeNode):
    """Rollout data needed in the outer loop for LP estimation.

    Shapes when returned from collect_data_and_update_agent:
    - raw_rewards: [U, T, B, R]
    - done_masks: [U, T, B, R]
    """

    raw_rewards: jax.Array
    done_masks: jax.Array


class AlphaScoreReplayBuffer(struct.PyTreeNode):
    """Fixed-size ring buffer storing (alpha, score) targets for curriculum training.

    Shapes:
    - alpha: [C, R]
    - score: [C]
    - is_valid: [C]
    - write_index, size: scalar
    """

    alpha: jax.Array
    score: jax.Array
    is_valid: jax.Array
    write_index: jax.Array
    size: jax.Array
    capacity: int = struct.field(pytree_node=False)
    batch_size: int = struct.field(pytree_node=False)


class CurriculumState(struct.PyTreeNode):
    """Outer-loop curriculum state threaded across iterations."""

    alpha_score_replay_buffer: AlphaScoreReplayBuffer
    score_predictor_train_state: TrainState
    lp_normalization_stats: "LpNormalizationStats"
    num_batches_seen: jax.Array


class LpNormalizationStats(struct.PyTreeNode):
    """EMA return-scale statistics used for LP normalization.

    Shapes:
    - mean, second_moment, var, initialized: [R]
    """

    mean: jax.Array
    second_moment: jax.Array
    var: jax.Array
    initialized: jax.Array


class RunnerStateBase(struct.PyTreeNode):
    """Common training carry state."""

    rng: jax.Array
    agent_train_state: TrainState
    env_state: Any
    prev_obs: jax.Array
    prev_done: jax.Array
    reward_normalization_stats: RewardNormalizationStats


class RunnerStateTransformer(RunnerStateBase):
    """Runner state extension with Transformer-XL memory caches.

    Batched Shapes in training:
    - memories: [B, past_context_length, num_transformer_blocks, transformer_hidden_states_dim]
    - memories_mask: [B, num_attn_heads, 1, past_context_length + 1]
    - memories_mask_idx: [B]
    """

    memories: jnp.ndarray
    memories_mask: jnp.ndarray
    memories_mask_idx: jnp.ndarray
