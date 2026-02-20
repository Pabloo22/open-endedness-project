import jax
import jax.numpy as jnp
from flax import struct


class Transition_data_base(struct.PyTreeNode):
    # for ppo update
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array


class Transition_data_rnd(Transition_data_base):
    # transformer specific
    memories_mask: jnp.ndarray
    memories_indices: jnp.ndarray
    # rnd specific
    next_obs: jnp.ndarray
    intrinsic_reward: jnp.ndarray
    intrinsic_value: jnp.ndarray
    done_for_intrinsic: jnp.ndarray
