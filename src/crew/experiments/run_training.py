"""Training entrypoint for main algorithm experiments."""

import dataclasses
import time
from typing import Any

import flax
import jax
import orbax.checkpoint as orbax
from flax.training import orbax_utils

from crew.experiments.paths import build_trained_weights_path
from crew.main_algo.config import TrainConfig
from crew.main_algo.main_loop import full_training
from crew.main_algo.setups import set_up_for_training


def run_main_algo_training(config: TrainConfig, save_results: bool = True) -> dict[str, Any]:
    """Run one full main-algorithm training experiment and optionally save results."""
    (
        rng,
        env,
        env_params,
        agent_train_state,
        reward_normalization_stats,
        intrinsic_modules,
        intrinsic_states,
        curriculum_state,
    ) = set_up_for_training(config)

    print(f"Training main algorithm with seed {config.train_seed}")
    start_time = time.time()

    train_info = jax.block_until_ready(
        full_training(
            rng=rng,
            agent_train_state=agent_train_state,
            reward_normalization_stats=reward_normalization_stats,
            intrinsic_states=intrinsic_states,
            curriculum_state=curriculum_state,
            env=env,
            env_params=env_params,
            intrinsic_modules=intrinsic_modules,
            config=config,
        )
    )

    elapsed_minutes = (time.time() - start_time) / 60.0
    print(f"Done in {elapsed_minutes:.2f} min")

    if save_results:
        try:
            save_path = build_trained_weights_path(
                algorithm_id="main_algo",
                env_id=config.env_id,
                seed=config.train_seed,
                artifacts_root=config.artifacts_root,
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)

            train_config = flax.core.freeze(dataclasses.asdict(config))
            training_results = {
                "config": train_config,
                "agent_params": train_info["agent_state"].params,
                "intrinsic_states": train_info["intrinsic_states"],
                "curriculum_state": train_info["curriculum_state"],
                "reward_normalization_stats": train_info["reward_normalization_stats"],
                "metrics": train_info["metrics"],
            }

            orbax_checkpointer = orbax.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(training_results)
            orbax_checkpointer.save(save_path, training_results, save_args=save_args)
            print("Saved training results to", save_path)
        except Exception as e:
            print(f"Error while saving training results to disk: {e}")

    return train_info


if __name__ == "__main__":
    # Smoke-friendly local run configuration.
    config = TrainConfig(
        train_seed=1,
        total_timesteps=100_000,
        env_id="Craftax-Classic-Symbolic-v1",
        num_envs_per_batch=64,
        num_steps_per_env=512,
        num_steps_per_update=256,
        update_epochs=1,
        num_minibatches=16,
        past_context_length=64,
        subsequence_length_in_loss_calculation=64,
        num_transformer_blocks=1,
        transformer_hidden_states_dim=64,
        qkv_features=64,
        head_hidden_dim=64,
    )
    run_main_algo_training(config=config, save_results=False)
