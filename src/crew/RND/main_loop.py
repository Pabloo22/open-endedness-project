from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.train_state import TrainState
from jax.tree_util import Partial

from crew.RND.config import TrainConfig
from crew.RND.data_collection_and_updates import collect_data_and_update
from crew.evaluations.rollouts import eval_rollout
from crew.RND.normalization_utils import NormalizationStats


def full_training_on_fixed_envs(
    rng: jax.Array,
    agent_train_state: TrainState,
    predictor_train_state: TrainState,
    target_train_state: TrainState,
    env: Any,
    env_params: Any,
    config: TrainConfig,
):
    ####   SETUP   ####
    rng, reset_rng_base = jax.random.split(rng, num=2)
    reset_rng = jax.random.split(reset_rng_base, num=config.num_envs_per_batch)
    prev_obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    prev_done = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.bool_)

    memories = jnp.zeros(
        (
            config.num_envs_per_batch,
            config.past_context_length,
            config.num_transformer_blocks,
            config.transformer_hidden_states_dim,
        )
    )
    memories_mask = jnp.zeros(
        (
            config.num_envs_per_batch,
            config.num_attn_heads,
            1,
            config.past_context_length + 1,
        ),
        dtype=jnp.bool_,
    )
    memories_mask_idx = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32) + (
        config.past_context_length + 1
    )

    normalization_stats = NormalizationStats(
        running_forward_return=jnp.zeros(config.num_envs_per_batch)
    )

    runner_state = (
        rng,
        agent_train_state,
        predictor_train_state,
        target_train_state,
        normalization_stats,
        prev_obs,
        env_state,
        prev_done,
        memories,
        memories_mask,
        memories_mask_idx,
        0,
    )

    def train_on_fixed_envs_one_iteration(
        runner_state,
        _unused,
        env: Any,
        env_params: Any,
        config: TrainConfig,
    ):
        # Train
        runner_state, metrics = jax.lax.scan(
            Partial(collect_data_and_update, env=env, env_params=env_params, config=config),
            runner_state,
            None,
            config.num_updates_per_batch,
        )

        metrics = jtu.tree_map(lambda x: x.mean(-1), metrics)
        rng, agent_train_state = runner_state[:2]

        # Evaluation
        rng, eval_rng_base = jax.random.split(rng)
        eval_rng = jax.random.split(eval_rng_base, num=config.eval_num_envs)
        _, eval_stats = jax.vmap(
            Partial(
                eval_rollout,
                env=env,
                env_params=env_params,
                train_state=agent_train_state,
                num_consecutive_episodes=config.eval_num_episodes,
                config=config,
            )
        )(eval_rng)

        # adding evaluation metrics
        metrics.update(
            {
                "eval/returns": eval_stats.returns,
                "eval/lengths": eval_stats.lengths,
                "lr": agent_train_state.opt_state[-1].hyperparams["learning_rate"],
            }
        )
        runner_state = (rng, *runner_state[1:])
        return runner_state, metrics

    final_runner_state, metrics = jax.lax.scan(
        Partial(
            train_on_fixed_envs_one_iteration,
            env=env,
            env_params=env_params,
            config=config,
        ),
        runner_state,
        None,
        config.num_batches_of_envs,
    )

    return {
        "agent_state": final_runner_state[1],
        "predictor_state": final_runner_state[2],
        "metrics": metrics,
    }
