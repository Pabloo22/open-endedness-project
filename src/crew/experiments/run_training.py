import dataclasses
import time
from typing import Any

import flax
import jax
import orbax.checkpoint as orbax
from flax.training import orbax_utils

from crew.experiments.paths import build_trained_weights_path
from crew.main_algo.baseline_main_loop import full_training_baseline
from crew.main_algo.config import TrainConfig
from crew.main_algo.main_loop import full_training
from crew.main_algo.setups import set_up_for_training


def run_main_algo_training(config: TrainConfig, save_results: bool = True) -> dict[str, Any]:
    """Run one full training experiment and optionally save results."""
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

    print(f"Training with seed {config.train_seed}")
    start_time = time.time()

    if config.training_mode == "curriculum":
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
    else:
        train_info = jax.block_until_ready(
            full_training_baseline(
                rng=rng,
                agent_train_state=agent_train_state,
                reward_normalization_stats=reward_normalization_stats,
                intrinsic_states=intrinsic_states,
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
            save_path = build_trained_weights_path(config=config)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            train_config = flax.core.freeze(dataclasses.asdict(config))
            training_results = {
                "config": train_config,
                "agent_params": train_info["agent_state"].params,
                "intrinsic_states": train_info["intrinsic_states"],
                "reward_normalization_stats": train_info["reward_normalization_stats"],
                "metrics": train_info["metrics"],
            }
            if "curriculum_state" in train_info:
                training_results["curriculum_state"] = train_info["curriculum_state"]

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
        achievement_ids_to_block=tuple(range(15)),  # Block the first n achievements for testing.
        training_mode="curriculum",  # "curriculum" or "baseline"
        selected_intrinsic_modules=("rnd",),
        baseline_fixed_training_alpha=(0.8, 0.2),  # Only used in baseline mode.
        num_envs_per_batch=64,
        num_steps_per_env=512,
        num_steps_per_update=256,
        eval_every_n_batches=2,
        eval_num_envs=64,
        eval_num_episodes=2,
        evaluation_alphas=((0.8, 0.2), (1.0, 0.0)),  # Only used in curriculum mode.
        update_epochs=1,
        num_minibatches=16,
        past_context_length=64,
        subsequence_length_in_loss_calculation=64,
        num_transformer_blocks=1,
        transformer_hidden_states_dim=64,
        qkv_features=64,
        head_hidden_dim=64,
        enable_wandb=True,
        is_timing_run=False,
    )
    run_main_algo_training(config=config, save_results=False)
