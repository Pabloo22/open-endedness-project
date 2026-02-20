from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState

from crew.RND.config import TrainConfig
from crew.RND.rnd_transformer_actor_critic import ActorCriticTransformer


class RolloutEpisodeStats(struct.PyTreeNode):
    returns: jax.Array
    lengths: jax.Array


def eval_rollout(
    rng: jax.Array,
    env: Any,
    env_params: Any,
    train_state: TrainState,
    num_consecutive_episodes: int,
    config: TrainConfig,
) -> tuple[jax.Array, RolloutEpisodeStats]:
    """Run one policy worker for `num_consecutive_episodes` episodes.

    Shapes:
    - returns: [num_consecutive_episodes]
    - lengths: [num_consecutive_episodes]
    """
    target_episodes = jnp.array(num_consecutive_episodes, dtype=jnp.int32)
    episode_stats = RolloutEpisodeStats(
        returns=jnp.zeros((num_consecutive_episodes,), dtype=jnp.float32),
        lengths=jnp.zeros((num_consecutive_episodes,), dtype=jnp.int32),
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
        )
    )
    memories_mask = jnp.zeros((1, config.num_attn_heads, 1, config.past_context_length + 1), dtype=jnp.bool_)
    memories_mask_idx = jnp.zeros((1,), dtype=jnp.int32) + (config.past_context_length + 1)

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
        _rng, _episode_stats, ep_num, _ep_len, _ep_ret, _obs, _env_state, _done_prev, _memories, _memories_mask, _memories_mask_idx = carry
        return ep_num < target_episodes

    def _body_fn(carry):
        rng, episode_stats, ep_num, ep_len, ep_ret, prev_obs, env_state, done_prev, memories, memories_mask, memories_mask_idx = carry

        memories_mask_idx = jnp.where(
            done_prev,
            config.past_context_length,
            jnp.clip(memories_mask_idx - 1, 0, config.past_context_length),
        )
        memories_mask = jnp.where(
            done_prev,  # done_prev[None, None, None, None]
            jnp.zeros_like(memories_mask),
            memories_mask,
        )
        memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config.past_context_length + 1)
        memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(config.num_attn_heads, 1)
        memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

        rng, action_rng, step_rng = jax.random.split(rng, num=3)
        pi, _, _, memories_out = train_state.apply_fn(
            train_state.params,
            memories,
            prev_obs[None, None, :],
            memories_mask,
            method=ActorCriticTransformer.model_forward_eval,
        )
        action = pi.sample(seed=action_rng).squeeze()
        memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

        obs, env_state, reward, done, _ = env.step(step_rng, env_state, action, env_params)

        ep_len = ep_len + 1
        ep_ret = ep_ret + reward

        def write_episode(carry_episode):
            episode_stats, ep_num, ep_len, ep_ret = carry_episode
            episode_stats = episode_stats.replace(
                returns=episode_stats.returns.at[ep_num].set(ep_ret),
                lengths=episode_stats.lengths.at[ep_num].set(ep_len),
            )
            return (
                episode_stats,
                ep_num + 1,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0.0, dtype=jnp.float32),
            )

        def do_nothing(carry_episode):
            return carry_episode

        episode_stats, ep_num, ep_len, ep_ret = jax.lax.cond(
            done,
            write_episode,
            do_nothing,
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
    rng, episode_stats = final_carry[0], final_carry[1]
    return rng, episode_stats
