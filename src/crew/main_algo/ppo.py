import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from crew.main_algo.actor_critic import ActorCriticTransformer
from crew.main_algo.types import TransitionDataTransformer


# Utility functions
def indices_select(x, y):
    return x[y]


batch_indices_select = jax.vmap(indices_select)
roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)


def batchify(x):
    return jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:]))


##############---------------------------------------


def update_epoch(update_state, _unused, config):
    def update_minibatch(train_state, batch_info):

        def loss_fn(params, transitions, memories_batch, alpha_batch, advantages, targets):
            # Key input shapes:
            # - transitions.obs: [B_mb, T, obs_dim]
            # - transitions.value, targets: [B_mb, T, R]
            # - advantages: [B_mb, T]
            # - alpha_batch: [B_mb, R]
            # - memories_batch: (B_mb, T + past_context_length, num_tranformer_layers, hidden_dim)

            subsequence_length = config.subsequence_length_in_loss_calculation
            num_subsequences = transitions.obs.shape[1] // subsequence_length

            # Gather cached memories at the first step of each subsequence.
            memories_batch = batch_indices_select(
                memories_batch,
                transitions.memories_indices[:, ::subsequence_length],
            )
            memories_batch = batchify(memories_batch)

            # Build causal masks for each subsequence.
            memories_mask = transitions.memories_mask.reshape(
                (-1, subsequence_length, *transitions.memories_mask.shape[2:]),
            )
            memories_mask = jnp.swapaxes(memories_mask, 1, 2)
            memories_mask = jnp.concatenate(
                (
                    memories_mask,
                    jnp.zeros(
                        (*memories_mask.shape[:-1], subsequence_length - 1),
                        dtype=jnp.bool_,
                    ),
                ),
                axis=-1,
            )
            memories_mask = roll_vmap(
                memories_mask,
                jnp.arange(0, subsequence_length),
                -1,
            )

            # [B_mb, T, ...] -> [B_mb * (T / subseq_len), subseq_len, ...]
            transitions, targets, advantages = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, subsequence_length, *x.shape[2:])),
                (transitions, targets, advantages),
            )
            alpha_batch = jnp.repeat(alpha_batch, repeats=num_subsequences, axis=0)  # [B * num_subsequences, R]

            pi, predicted_values = train_state.apply_fn(
                params,
                memories_batch,
                transitions.obs,
                memories_mask,
                alpha_batch,
                method=ActorCriticTransformer.model_forward_train,
            )
            log_prob = pi.log_prob(transitions.action)

            # Critic loss over all reward-function heads.
            values_pred_clipped = transitions.value + (predicted_values - transitions.value).clip(
                -config.clip_eps, config.clip_eps
            )
            value_losses = jnp.square(predicted_values - targets)
            value_losses_clipped = jnp.square(values_pred_clipped - targets)
            value_loss_per_reward_function = 0.5 * jnp.mean(
                jnp.maximum(value_losses, value_losses_clipped),
                axis=(0, 1),
            )  # [R]
            value_loss_total = jnp.mean(value_loss_per_reward_function)

            # Actor loss from precomputed alpha-weighted advantages.
            log_ratio = log_prob - transitions.log_prob
            ratio = jnp.exp(log_ratio)
            actor_loss_unclipped = ratio * advantages
            actor_loss_clipped = (
                jnp.clip(
                    ratio,
                    1.0 - config.clip_eps,
                    1.0 + config.clip_eps,
                )
                * advantages
            )
            actor_loss = -jnp.minimum(actor_loss_unclipped, actor_loss_clipped).mean()

            entropy = pi.entropy().mean()
            total_loss = actor_loss + config.vf_coef * value_loss_total - config.ent_coef * entropy
            approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
            return total_loss, (value_loss_per_reward_function, actor_loss, entropy, approx_kl)

        transitions, memories_batch, alpha_batch, advantages, targets = batch_info

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (value_loss_per_reward_function, actor_loss, entropy, approx_kl)), grads = grad_fn(
            train_state.params,
            transitions,
            memories_batch,
            alpha_batch,
            advantages,
            targets,
        )
        train_state = train_state.apply_gradients(grads=grads)
        metrics = {
            "ppo/total_loss": total_loss,
            "ppo/value_loss": value_loss_per_reward_function,
            "ppo/actor_loss": actor_loss,
            "ppo/entropy": entropy,
            "ppo/approx_kl": approx_kl,
        }
        return train_state, metrics

    rng, train_state, transitions, memories_batch, alpha_batch, advantages, targets = update_state

    # Shuffle on environment axis and split into minibatches.
    rng, shuffle_rng = jax.random.split(rng)
    permutation = jax.random.permutation(shuffle_rng, config.num_envs_per_batch)

    intermediate = (transitions, memories_batch, advantages, targets)
    intermediate = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), intermediate)
    minibatches = (intermediate[0], intermediate[1], alpha_batch, intermediate[2], intermediate[3])
    minibatches = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), minibatches)
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (config.num_minibatches, -1, *x.shape[1:])),
        minibatches,
    )

    train_state, metrics = jax.lax.scan(update_minibatch, train_state, minibatches)

    update_state = (
        rng,
        train_state,
        transitions,
        memories_batch,
        alpha_batch,
        advantages,
        targets,
    )
    return update_state, metrics


def update_agent(
    rng: jax.Array,
    agent_train_state,
    transitions: TransitionDataTransformer,
    memories_batch: jax.Array,
    alpha_batch: jax.Array,
    weighted_advantages: jax.Array,
    value_targets: jax.Array,
    config,
):
    """Run PPO epochs/minibatches for one collected rollout window."""
    update_state = (
        rng,
        agent_train_state,
        transitions,
        memories_batch,
        alpha_batch,
        weighted_advantages,
        value_targets,
    )
    update_state, metrics = jax.lax.scan(
        Partial(update_epoch, config=config),
        update_state,
        None,
        config.update_epochs,
    )
    rng, agent_train_state = update_state[:2]
    metrics = jax.tree_util.tree_map(lambda x: x.mean(axis=(0, 1)), metrics)
    return rng, agent_train_state, metrics
