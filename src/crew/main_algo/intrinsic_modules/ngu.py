"""NGU intrinsic reward module implementation for the main algorithm."""

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from crew.main_algo.types import (
    IntrinsicModulesUpdateData,
    IntrinsicUpdateMetrics,
    TransitionDataBase,
)
from crew.networks.encoders import ObsEncoderFlatSymbolic


class NGUEmbeddingNetwork(nn.Module):
    encoder_mode: str
    output_embedding_dim: int
    obs_emb_dim: int
    head_activation: str
    head_hidden_dim: int

    def setup(self):
        self.input_encoder = ObsEncoderFlatSymbolic(obs_emb_dim=self.obs_emb_dim)

        if self.head_activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        self.linear1 = nn.Dense(
            self.head_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.linear_out = nn.Dense(
            self.output_embedding_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )

    def __call__(self, observations: jax.Array) -> jax.Array:
        encoded_input = self.input_encoder(observations=observations)
        outputs = self.linear1(encoded_input)
        outputs = self.activation_fn(outputs)
        return self.linear_out(outputs)


class NGUModuleState(struct.PyTreeNode):
    """NGU module state carried by the training loop."""

    embedding_train_state: TrainState
    episodic_memory: jax.Array  # [B, capacity, emb_dim] — per-env memory buffer
    memory_size: jax.Array  # [B] — how many slots filled per env


def _run_episodic_memory_scan(
    embedding_train_state: TrainState,
    initial_memory: jax.Array,
    initial_memory_size: jax.Array,
    obs: jax.Array,
    done: jax.Array,
    capacity: int,
    num_neighbors: int,
    kernel_epsilon: float,
    kernel_cluster_distance: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Scan over T time steps, filling episodic memory and computing kNN bonus.

    Args:
        obs: [T, B, obs_dim]
        done: [T, B]
        initial_memory: [B, capacity, emb_dim]
        initial_memory_size: [B]

    Returns:
        rewards: [T, B]
        final_memory: [B, capacity, emb_dim]
        final_memory_size: [B]
    """

    def step_fn(carry, inputs):
        memory, memory_size = carry  # [B, C, D], [B]
        obs_t, done_t = inputs  # [B, obs_dim], [B]

        # Embed obs and L2-normalize.
        embedding = embedding_train_state.apply_fn(
            embedding_train_state.params, obs_t
        )  # [B, D]
        norm = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
        embedding = embedding / (norm + 1e-8)

        # Squared L2 distances from each env's embedding to its memory: [B, C].
        dists = jnp.sum((memory - embedding[:, None, :]) ** 2, axis=-1)

        # Mask slots beyond memory_size[b] with inf.
        slots = jnp.arange(capacity)  # [C]
        valid_mask = slots[None, :] < memory_size[:, None]  # [B, C]
        dists = jnp.where(valid_mask, dists, jnp.inf)

        # Top-k nearest + kNN kernel → episodic bonus [B].
        topk = jnp.sort(dists, axis=-1)[:, :num_neighbors]  # [B, k]
        kernel = kernel_epsilon / (topk + kernel_epsilon)
        reward = jnp.sqrt(jnp.sum(kernel, axis=-1)) + kernel_cluster_distance  # [B]

        # Write embedding into memory at circular index.
        write_idx = memory_size % capacity  # [B]
        new_memory = jax.vmap(lambda mem, emb, idx: mem.at[idx].set(emb))(
            memory, embedding, write_idx
        )
        new_memory_size = jnp.minimum(memory_size + 1, capacity)

        # Reset memory and size where episode ended.
        new_memory = jnp.where(
            done_t[:, None, None], jnp.zeros_like(new_memory), new_memory
        )
        new_memory_size = jnp.where(
            done_t, jnp.zeros_like(new_memory_size), new_memory_size
        )

        return (new_memory, new_memory_size), reward

    (final_memory, final_memory_size), rewards = jax.lax.scan(
        step_fn,
        (initial_memory, initial_memory_size),
        (obs, done),
    )
    return rewards, final_memory, final_memory_size


class NGUIntrinsicModule:
    """NGU intrinsic reward module."""

    name = "ngu"
    is_episodic = True

    def init_state(
        self,
        rng: jax.Array,
        obs_shape: tuple[int, ...],
        config: Any,
    ) -> NGUModuleState:
        init_obs = jnp.zeros((2, *obs_shape), dtype=jnp.float32)
        network = NGUEmbeddingNetwork(
            encoder_mode=config.ngu.encoder_mode,
            output_embedding_dim=config.ngu.output_embedding_dim,
            obs_emb_dim=config.obs_emb_dim,
            head_activation=config.ngu.head_activation,
            head_hidden_dim=config.ngu.head_hidden_dim,
        )
        params = network.init(rng, init_obs)
        tx = optax.chain(
            optax.inject_hyperparams(optax.adam)(
                learning_rate=config.ngu.embedding_network_lr,
                eps=config.adam_eps,
            ),
        )
        embedding_train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )
        episodic_memory = jnp.zeros(
            (
                config.num_envs_per_batch,
                config.ngu.episodic_memory_capacity,
                config.ngu.output_embedding_dim,
            ),
            dtype=jnp.float32,
        )
        memory_size = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32)
        return NGUModuleState(
            embedding_train_state=embedding_train_state,
            episodic_memory=episodic_memory,
            memory_size=memory_size,
        )

    def compute_rewards(
        self,
        rng: jax.Array,
        module_state: NGUModuleState,
        transitions: TransitionDataBase,
        config: Any,
    ) -> jax.Array:
        del rng
        rewards, _, _ = _run_episodic_memory_scan(
            embedding_train_state=module_state.embedding_train_state,
            initial_memory=module_state.episodic_memory,
            initial_memory_size=module_state.memory_size,
            obs=transitions.obs,
            done=transitions.done,
            capacity=config.ngu.episodic_memory_capacity,
            num_neighbors=config.ngu.num_neighbors,
            kernel_epsilon=config.ngu.kernel_epsilon,
            kernel_cluster_distance=config.ngu.kernel_cluster_distance,
        )
        return rewards

    def update(
        self,
        rng: jax.Array,
        module_state: NGUModuleState,
        transitions: IntrinsicModulesUpdateData,
        config: Any,
    ) -> tuple[NGUModuleState, IntrinsicUpdateMetrics]:
        del rng
        # First pass: frozen embedding — no gradient update, but persist the
        # final memory state so episodes that span multiple rollouts carry over.
        _, final_memory, final_memory_size = _run_episodic_memory_scan(
            embedding_train_state=module_state.embedding_train_state,
            initial_memory=module_state.episodic_memory,
            initial_memory_size=module_state.memory_size,
            obs=transitions.obs,
            done=transitions.done,
            capacity=config.ngu.episodic_memory_capacity,
            num_neighbors=config.ngu.num_neighbors,
            kernel_epsilon=config.ngu.kernel_epsilon,
            kernel_cluster_distance=config.ngu.kernel_cluster_distance,
        )
        updated_state = module_state.replace(
            episodic_memory=final_memory,
            memory_size=final_memory_size,
        )
        metrics: IntrinsicUpdateMetrics = {
            "intrinsic_modules/ngu/embedding_loss": jnp.array(0.0),
        }
        return updated_state, metrics

    def done_mask(self, env_done: jax.Array, config: Any) -> jax.Array:  # noqa: ARG002
        del config
        return env_done
