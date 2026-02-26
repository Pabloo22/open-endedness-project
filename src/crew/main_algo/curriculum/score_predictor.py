"""Score predictor model and training utilities for curriculum sampling."""

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from crew.main_algo.types import AlphaScoreReplayBuffer


class ScorePredictorMLP(nn.Module):
    """Small MLP used to predict score from alpha vectors."""

    hidden_dim: int
    activation: str

    def setup(self):
        if self.activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        self.linear1 = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.linear2 = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.linear_out = nn.Dense(1, kernel_init=nn.initializers.variance_scaling(1e-2, "fan_in", "truncated_normal"), bias_init=nn.initializers.zeros)

    def __call__(self, alpha: jax.Array) -> jax.Array:
        """Predict scalar scores for alpha inputs.

        Shapes (in codebase usage) are:
        - alpha: [N, R]
        - returns: [N]
        """
        outputs = self.linear1(alpha)
        outputs = self.activation_fn(outputs)
        outputs = self.linear2(outputs)
        outputs = self.activation_fn(outputs)
        return self.linear_out(outputs).squeeze(axis=-1)


def init_score_predictor_train_state(
    rng: jax.Array,
    config: Any,
) -> TrainState:
    """Initialize score predictor train state."""
    model = ScorePredictorMLP(
        hidden_dim=config.curriculum.predictor_hidden_dim,
        activation=config.curriculum.predictor_activation,
    )
    init_alpha = jnp.zeros((2, config.num_reward_functions), dtype=jnp.float32)
    model_params = model.init(rng, init_alpha)
    tx = optax.chain(
        optax.inject_hyperparams(optax.adam)(
            learning_rate=config.curriculum.predictor_lr,
            eps=config.adam_eps,
        ),
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=model_params,
        tx=tx,
    )


def train_score_predictor_on_buffer(
    rng: jax.Array,
    score_predictor_train_state: TrainState,
    alpha_score_replay_buffer: AlphaScoreReplayBuffer,
    config: Any,
) -> tuple[jax.Array, TrainState, dict[str, jax.Array]]:
    """Train score predictor on replay buffer data with masked MSE.

    Input shapes:
    - alpha_score_replay_buffer.alpha: [C, R]
    - alpha_score_replay_buffer.score: [C]
    - alpha_score_replay_buffer.is_valid: [C]
    """
    alpha_data = alpha_score_replay_buffer.alpha
    score_targets = alpha_score_replay_buffer.score
    valid_mask = alpha_score_replay_buffer.is_valid.astype(jnp.float32)

    num_minibatches = config.curriculum.predictor_num_minibatches
    total_num_datapoints = alpha_data.shape[0]
    minibatch_size = total_num_datapoints // num_minibatches

    def loss_fn(
        predictor_params: Any,
        alpha_chunk: jax.Array,
        target_chunk: jax.Array,
        valid_chunk: jax.Array,
    ) -> jax.Array:
        predicted_scores = score_predictor_train_state.apply_fn(predictor_params, alpha_chunk)  # [mb]
        squared_errors = jnp.square(predicted_scores - target_chunk)  # [mb]
        masked_errors = squared_errors * valid_chunk  # [mb]
        normalization = jnp.maximum(jnp.sum(valid_chunk), jnp.array(1.0, dtype=valid_chunk.dtype))
        return jnp.sum(masked_errors) / normalization

    def epoch_step(
        carry: tuple[jax.Array, TrainState],
        _unused: None,
    ) -> tuple[tuple[jax.Array, TrainState], jax.Array]:
        rng_epoch, train_state = carry
        rng_epoch, shuffle_rng = jax.random.split(rng_epoch)
        shuffled_indices = jax.random.permutation(shuffle_rng, total_num_datapoints)

        alpha_shuffled = jnp.take(alpha_data, shuffled_indices, axis=0)
        targets_shuffled = jnp.take(score_targets, shuffled_indices, axis=0)
        mask_shuffled = jnp.take(valid_mask, shuffled_indices, axis=0)

        alpha_shuffled = alpha_shuffled.reshape((num_minibatches, minibatch_size, *alpha_shuffled.shape[1:]))
        targets_shuffled = targets_shuffled.reshape((num_minibatches, minibatch_size))
        mask_shuffled = mask_shuffled.reshape((num_minibatches, minibatch_size))

        def minibatch_step(
            minibatch_train_state: TrainState,
            minibatch_inputs: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[TrainState, jax.Array]:
            alpha_chunk, targets_chunk, mask_chunk = minibatch_inputs
            loss_value, grads = jax.value_and_grad(loss_fn)(
                minibatch_train_state.params,
                alpha_chunk,
                targets_chunk,
                mask_chunk,
            )
            minibatch_train_state = minibatch_train_state.apply_gradients(grads=grads)
            return minibatch_train_state, loss_value

        train_state, minibatch_losses = jax.lax.scan(
            minibatch_step,
            train_state,
            (alpha_shuffled, targets_shuffled, mask_shuffled),
        )
        epoch_loss = jnp.mean(minibatch_losses)
        return (rng_epoch, train_state), epoch_loss

    (rng, score_predictor_train_state), epoch_losses = jax.lax.scan(
        epoch_step,
        (rng, score_predictor_train_state),
        None,
        config.curriculum.predictor_update_epochs,
    )

    diagnostics = {
        "curriculum/predictor_loss": jnp.mean(epoch_losses),
    }
    return rng, score_predictor_train_state, diagnostics
