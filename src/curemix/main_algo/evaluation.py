"""Evaluation utilities for main_algo."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState

from curemix.main_algo.actor_critic import ActorCriticTransformer


class EvalStats(struct.PyTreeNode):
    """Container for evaluation outputs."""

    returns: jax.Array
    lengths: jax.Array
    achievements: jax.Array


def infer_achievement_names(env: Any, env_params: Any) -> tuple[str, ...]:
    """Infer the ordered achievement info keys from one environment step."""
    rng = jax.random.key(0)
    rng, reset_rng, step_rng = jax.random.split(rng, num=3)
    _, env_state = env.reset(reset_rng, env_params)
    dummy_action = jnp.array(0, dtype=jnp.int32)
    _, _, _, _, info = env.step(step_rng, env_state, dummy_action, env_params)

    achievement_names = tuple(sorted(key for key in info if key.startswith("Achievements/")))
    if not achievement_names:
        msg = "No achievement keys found in env.step info with prefix 'Achievements/'."
        raise ValueError(msg)
    return achievement_names


def _extract_achievement_vector(info: dict[str, jax.Array], achievement_names: tuple[str, ...]) -> jax.Array:
    """Extract terminal-episode achievements as a binary vector [num_achievements]."""
    achievement_values = [jnp.asarray(info[name]) > 0.0 for name in achievement_names]
    return jnp.asarray(achievement_values, dtype=jnp.bool_)


def _eval_rollout_single_env(
    rng: jax.Array,
    alpha: jax.Array,
    train_state: TrainState,
    env: Any,
    env_params: Any,
    num_eval_episodes: int,
    achievement_names: tuple[str, ...],
    config: Any,
) -> tuple[jax.Array, EvalStats]:
    """Roll out one env for a fixed alpha until `num_eval_episodes` episodes finish.

    Shapes:
    - alpha: [R]
    - returns: [num_eval_episodes]
    - lengths: [num_eval_episodes]
    - achievements: [num_eval_episodes, num_achievements]
    """
    target_episodes = jnp.array(num_eval_episodes, dtype=jnp.int32)
    num_achievements = len(achievement_names)
    episode_stats = EvalStats(
        returns=jnp.zeros((num_eval_episodes,), dtype=jnp.float32),
        lengths=jnp.zeros((num_eval_episodes,), dtype=jnp.int32),
        achievements=jnp.zeros((num_eval_episodes, num_achievements), dtype=jnp.bool_),
    )

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)
    done_prev = jnp.array(False, dtype=jnp.bool_)

    ep_len = jnp.array(0, dtype=jnp.int32)
    ep_ret = jnp.array(0.0, dtype=jnp.float32)
    ep_num = jnp.array(0, dtype=jnp.int32)

    memories = jnp.zeros(
        (
            1,
            config.past_context_length,
            config.num_transformer_blocks,
            config.transformer_hidden_states_dim,
        ),
        dtype=jnp.float32,
    )
    memories_mask = jnp.zeros(
        (1, config.num_attn_heads, 1, config.past_context_length + 1),
        dtype=jnp.bool_,
    )
    memories_mask_idx = jnp.full(
        (1,),
        config.past_context_length + 1,
        dtype=jnp.int32,
    )
    init_carry = (
        rng,
        episode_stats,
        ep_num,
        ep_len,
        ep_ret,
        obs,
        env_state,
        done_prev,
        memories,
        memories_mask,
        memories_mask_idx,
    )

    def _cond_fn(carry):
        return carry[2] < target_episodes

    def _body_fn(carry):
        (
            rng,
            episode_stats,
            ep_num,
            ep_len,
            ep_ret,
            prev_obs,
            env_state,
            done_prev,
            memories,
            memories_mask,
            memories_mask_idx,
        ) = carry

        # Advance attention mask for the next observation; reset mask when previous step ended an episode.
        memories_mask_idx = jnp.where(
            done_prev,
            config.past_context_length,
            jnp.clip(memories_mask_idx - 1, 0, config.past_context_length),
        )
        memories_mask = jnp.where(done_prev, jnp.zeros_like(memories_mask), memories_mask)
        memories_mask_idx_one_hot = jax.nn.one_hot(
            memories_mask_idx,
            config.past_context_length + 1,
            dtype=jnp.bool_,
        )
        memories_mask_idx_one_hot = jnp.repeat(
            memories_mask_idx_one_hot[:, None, None, :],
            config.num_attn_heads,
            axis=1,
        )
        memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_one_hot)

        # Sample one action from the alpha-conditioned policy.
        rng, action_rng, step_rng = jax.random.split(rng, num=3)
        alpha_batch = alpha[None, :]
        pi, _, memories_out = train_state.apply_fn(
            train_state.params,
            memories,
            prev_obs[None, None, :],
            memories_mask,
            alpha_batch,
            method=ActorCriticTransformer.model_forward_eval,
        )
        action = pi.sample(seed=action_rng).squeeze(axis=0)
        memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

        # Step the environment and accumulate episodic return/length.
        obs, env_state, reward, done, info = env.step(
            step_rng,
            env_state,
            action,
            env_params,
        )

        ep_len = ep_len + 1
        ep_ret = ep_ret + reward

        # On terminal transitions, write one episode entry and reset counters.
        def write_episode(carry_episode):
            current_episode_stats, current_ep_num, current_ep_len, current_ep_ret = carry_episode
            achievements_completion = _extract_achievement_vector(info, achievement_names)
            current_episode_stats = current_episode_stats.replace(
                returns=current_episode_stats.returns.at[current_ep_num].set(current_ep_ret),
                lengths=current_episode_stats.lengths.at[current_ep_num].set(current_ep_len),
                achievements=current_episode_stats.achievements.at[current_ep_num].set(achievements_completion),
            )
            return (
                current_episode_stats,
                current_ep_num + 1,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0.0, dtype=jnp.float32),
            )

        episode_stats, ep_num, ep_len, ep_ret = jax.lax.cond(
            done,
            write_episode,
            lambda carry_episode: carry_episode,
            (episode_stats, ep_num, ep_len, ep_ret),
        )

        return (
            rng,
            episode_stats,
            ep_num,
            ep_len,
            ep_ret,
            obs,
            env_state,
            done,
            memories,
            memories_mask,
            memories_mask_idx,
        )

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_carry)
    return final_carry[0], final_carry[1]


def evaluate_policy_on_alphas(
    rng: jax.Array,
    train_state: TrainState,
    env: Any,
    env_params: Any,
    evaluation_alphas: jax.Array,
    num_eval_envs: int,
    num_eval_episodes: int,
    achievement_names: tuple[str, ...],
    config: Any,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Evaluate policy across fixed alpha vectors.

    Shapes:
    - evaluation_alphas: [num_alphas, R]
    - returns: [num_alphas, B, num_eval_episodes]
    - lengths: [num_alphas, B, num_eval_episodes]
    - achievements: [num_alphas, B, num_eval_episodes, num_achievements]
    """
    num_alphas = evaluation_alphas.shape[0]
    next_rng, alpha_rng_base = jax.random.split(rng)
    alpha_rngs = jax.random.split(alpha_rng_base, num=num_alphas)

    def _evaluate_single_alpha(alpha_rng: jax.Array, alpha: jax.Array):
        alpha_rng, env_rng_base = jax.random.split(alpha_rng)
        env_rngs = jax.random.split(env_rng_base, num=num_eval_envs)
        alpha_batch = jnp.broadcast_to(alpha[None, :], (num_eval_envs, alpha.shape[0]))
        _, env_stats = jax.vmap(
            lambda env_rng, env_alpha: _eval_rollout_single_env(
                rng=env_rng,
                alpha=env_alpha,
                train_state=train_state,
                env=env,
                env_params=env_params,
                num_eval_episodes=num_eval_episodes,
                achievement_names=achievement_names,
                config=config,
            )
        )(env_rngs, alpha_batch)
        return alpha_rng, env_stats

    _, eval_env_stats = jax.vmap(_evaluate_single_alpha)(alpha_rngs, evaluation_alphas)
    return next_rng, {
        "eval/returns": eval_env_stats.returns,
        "eval/lengths": eval_env_stats.lengths,
        "eval/achievements": eval_env_stats.achievements,
    }
