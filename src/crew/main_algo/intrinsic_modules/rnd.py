"""RND intrinsic reward module implementation for the main algorithm."""

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
from crew.networks.encoders import build_observation_encoder


class RNDTargetAndPredictor(nn.Module):
    encoder_mode: str
    env_id: str
    output_embedding_dim: int
    obs_emb_dim: int
    head_activation: str
    head_hidden_dim: int
    use_inventory_only: bool = False

    def setup(self):
        if not self.use_inventory_only:
            self.input_encoder = build_observation_encoder(
                encoder_mode=self.encoder_mode,
                env_id=self.env_id,
                obs_emb_dim=self.obs_emb_dim,
            )
        else:
            self.input_encoder = None

        if self.head_activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        self.linear1 = nn.Dense(
            self.head_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.linear2 = nn.Dense(
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
        if self.use_inventory_only:
            from crew.networks.encoders import (
                CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
                CRAFTAX_CLASSIC_HEIGHT,
                CRAFTAX_CLASSIC_WIDTH,
                CRAFTAX_CLASSIC_MAP_CHANNELS,
                CRAFTAX_HEIGHT,
                CRAFTAX_WIDTH,
                CRAFTAX_MAP_CHANNELS,
            )

            if self.env_id == CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID:
                flat_map_dim = CRAFTAX_CLASSIC_HEIGHT * CRAFTAX_CLASSIC_WIDTH * CRAFTAX_CLASSIC_MAP_CHANNELS
                inventory_dim = 12
            else:
                flat_map_dim = CRAFTAX_HEIGHT * CRAFTAX_WIDTH * CRAFTAX_MAP_CHANNELS
                inventory_dim = 16

            # The observation is flat, map comes first, then extra features. The inventory is the first part of extra features.
            encoded_input = observations[..., flat_map_dim : flat_map_dim + inventory_dim]
        else:
            encoded_input = self.input_encoder(observations=observations)

        outputs = self.linear1(encoded_input)
        outputs = self.activation_fn(outputs)
        outputs = self.linear2(outputs)
        outputs = self.activation_fn(outputs)
        return self.linear_out(outputs)


class RNDModuleState(struct.PyTreeNode):
    """RND module state carried by the training loop."""

    predictor_train_state: TrainState
    target_train_state: TrainState


def _compute_intrinsic_rewards(
    predictor_train_state: TrainState,
    target_train_state: TrainState,
    transitions: TransitionDataBase,
    num_chunks: int,
) -> jax.Array:
    """Compute raw RND intrinsic rewards.

    Inputs:
    - transitions.next_obs: [T, B, *obs_shape]
    Outputs:
    - intrinsic_rewards: [T, B]
    """
    seq_len, batch_size = transitions.next_obs.shape[:2]
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (seq_len * batch_size, *x.shape[2:])),
        transitions,
    )
    total_num_steps = transitions.next_obs.shape[0]
    chunk_size = total_num_steps // num_chunks
    transitions_chunked = jax.tree_util.tree_map(
        lambda x: x.reshape((num_chunks, chunk_size, *x.shape[1:])),
        transitions,
    )

    def scan_step(carry, transitions_chunk):
        # next_obs chunk shape: [chunk_size, *obs_shape]
        target_embeddings = target_train_state.apply_fn(target_train_state.params, transitions_chunk.next_obs)
        predictor_embeddings = predictor_train_state.apply_fn(predictor_train_state.params, transitions_chunk.next_obs)
        prediction_errors_chunk = jnp.mean(jnp.square(target_embeddings - predictor_embeddings), axis=-1)
        return carry, prediction_errors_chunk

    _, prediction_errors_chunked = jax.lax.scan(scan_step, None, transitions_chunked)
    intrinsic_rewards = prediction_errors_chunked.reshape((total_num_steps,)).reshape((seq_len, batch_size))

    return intrinsic_rewards


def _train_predictor_network(
    rng: jax.Array,
    predictor_train_state: TrainState,
    target_train_state: TrainState,
    transitions: IntrinsicModulesUpdateData,
    num_epochs: int,
    num_minibatches: int,
) -> tuple[TrainState, jax.Array]:
    """Train RND predictor network on collected transitions.

    Inputs:
    - transitions.next_obs: [T, B, *obs_shape]
    Outputs:
    - updated predictor state
    - mean predictor loss (scalar)
    """
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1, *x.shape[2:])),
        transitions,
    )
    total_num_steps = transitions.next_obs.shape[0]
    minibatch_size = total_num_steps // num_minibatches

    def loss_fn(predictor_params, transitions_chunk):
        target_embeddings = target_train_state.apply_fn(target_train_state.params, transitions_chunk.next_obs)
        predictor_embeddings = predictor_train_state.apply_fn(predictor_params, transitions_chunk.next_obs)
        return jnp.mean(jnp.square(target_embeddings - predictor_embeddings))

    def epoch_step(carry, _unused):
        rng, train_state = carry
        rng, shuffle_rng = jax.random.split(rng)
        shuffled_indices = jax.random.permutation(shuffle_rng, total_num_steps)
        transitions_shuffled = jax.tree_util.tree_map(
            lambda x: jnp.take(x, shuffled_indices, axis=0),
            transitions,
        )
        transitions_shuffled = jax.tree_util.tree_map(
            lambda x: x.reshape((num_minibatches, minibatch_size, *x.shape[1:])),
            transitions_shuffled,
        )

        def minibatch_step(train_state, transitions_chunk):
            loss_value, grads = jax.value_and_grad(loss_fn)(train_state.params, transitions_chunk)
            new_train_state = train_state.apply_gradients(grads=grads)
            return new_train_state, loss_value

        train_state, batch_losses = jax.lax.scan(minibatch_step, train_state, transitions_shuffled)
        epoch_loss = jnp.mean(batch_losses)
        return (rng, train_state), epoch_loss

    (_, updated_predictor_state), epoch_losses = jax.lax.scan(
        epoch_step,
        (rng, predictor_train_state),
        None,
        num_epochs,
    )
    return updated_predictor_state, jnp.mean(epoch_losses)


class RNDIntrinsicModule:
    """RND intrinsic reward module with fixed done-mask semantics."""

    name = "rnd"
    is_episodic = False

    def init_state(
        self,
        rng: jax.Array,
        obs_shape: tuple[int, ...],
        config: Any,
    ) -> RNDModuleState:
        init_obs = jnp.zeros((2, *obs_shape), dtype=jnp.float32)
        network = RNDTargetAndPredictor(
            encoder_mode=config.encoder_mode,
            env_id=config.env_id,
            output_embedding_dim=config.rnd.output_embedding_dim,
            obs_emb_dim=config.obs_emb_dim,
            head_activation=config.rnd.head_activation,
            head_hidden_dim=config.rnd.head_hidden_dim,
            use_inventory_only=config.rnd.use_inventory_only,
        )

        predictor_rng, target_rng = jax.random.split(rng, 2)
        predictor_params = network.init(predictor_rng, init_obs)
        target_params = network.init(target_rng, init_obs)

        predictor_tx = optax.chain(
            optax.inject_hyperparams(optax.adam)(
                learning_rate=config.rnd.predictor_network_lr,
                eps=config.adam_eps,
            ),
        )
        predictor_train_state = TrainState.create(
            apply_fn=network.apply,
            params=predictor_params,
            tx=predictor_tx,
        )
        target_train_state = TrainState.create(
            apply_fn=network.apply,
            params=target_params,
            tx=optax.identity(),
        )
        return RNDModuleState(
            predictor_train_state=predictor_train_state,
            target_train_state=target_train_state,
        )

    def compute_rewards(
        self,
        rng: jax.Array,  # noqa: ARG002
        module_state: RNDModuleState,
        transitions: TransitionDataBase,
        config: Any,
    ) -> jax.Array:
        del rng
        # Returns raw intrinsic rewards; no normalization.
        return _compute_intrinsic_rewards(
            predictor_train_state=module_state.predictor_train_state,
            target_train_state=module_state.target_train_state,
            transitions=transitions,
            num_chunks=config.rnd.num_chunks_in_rewards_computation,
        )

    def update(
        self,
        rng: jax.Array,
        module_state: RNDModuleState,
        transitions: IntrinsicModulesUpdateData,
        config: Any,
    ) -> tuple[RNDModuleState, IntrinsicUpdateMetrics]:
        updated_predictor_state, mean_predictor_loss = _train_predictor_network(
            rng=rng,
            predictor_train_state=module_state.predictor_train_state,
            target_train_state=module_state.target_train_state,
            transitions=transitions,
            num_epochs=config.rnd.predictor_update_epochs,
            num_minibatches=config.rnd.predictor_num_minibatches,
        )
        updated_module_state = module_state.replace(predictor_train_state=updated_predictor_state)
        metrics: IntrinsicUpdateMetrics = {
            "intrinsic_modules/rnd/predictor_loss": mean_predictor_loss,
        }
        return updated_module_state, metrics

    def done_mask(self, env_done: jax.Array, config: Any) -> jax.Array:  # noqa: ARG002
        del config
        # Keep RND return accumulation uninterrupted across episode boundaries.
        return jnp.zeros_like(env_done, dtype=jnp.bool_)
