import argparse
import dataclasses
import time
from collections.abc import Sequence
from typing import Any

import flax
import jax
import orbax.checkpoint as orbax
from flax.training import orbax_utils

from crew.experiments.constants import CRAFTAX_CLASSIC_INTERMEDIATE_ACHIEVEMENT_IDS
from crew.experiments.paths import build_trained_weights_path
from crew.main_algo.baseline_main_loop import full_training_baseline
from crew.main_algo.config import TrainConfig
from crew.main_algo.main_loop import full_training
from crew.main_algo.setups import set_up_for_training


def build_smoke_run_config() -> TrainConfig:
    """Build the reduced-cost config used for local smoke runs."""
    return TrainConfig(
        train_seed=1,
        procedural_generation=False,
        fixed_reset_seed=101,  # Only used if procedural_generation=False.
        total_timesteps=10_000_000,
        env_id="Craftax-Classic-Symbolic-v1",
        achievement_ids_to_block=CRAFTAX_CLASSIC_INTERMEDIATE_ACHIEVEMENT_IDS,  # Block the first n achievements for testing.
        remove_health_reward=True,
        episode_max_steps=1024,
        training_mode="baseline",
        selected_intrinsic_modules=("rnd",),
        baseline_fixed_training_alpha=(0.8, 0.2),
        encoder_mode="craftax_structured",
        num_envs_per_batch=256,
        num_steps_per_env=2048,
        num_steps_per_update=256,
        eval_every_n_batches=3,
        eval_num_envs=128,
        eval_num_episodes=2,
        evaluation_alphas=((0.8, 0.2), (1.0, 0.0)),
        update_epochs=1,
        num_minibatches=16,
        lr=2e-4,
        clip_eps=0.2,
        gae_lambda=0.95,
        ent_coef=0.01,
        obs_emb_dim=128,
        past_context_length=64,
        subsequence_length_in_loss_calculation=32,
        num_attn_heads=4,
        num_transformer_blocks=1,
        transformer_hidden_states_dim=128,
        qkv_features=128,
        head_hidden_dim=128,
        enable_wandb=True,
        wandb_entity="openendedness-2026",
        is_timing_run=False,
    )


def build_training_run_config() -> TrainConfig:
    """Build the full training config used by this entrypoint."""
    return TrainConfig(
        train_seed=1,
        procedural_generation=False,
        total_timesteps=500_000_000,
        env_id="Craftax-Classic-Symbolic-v1",
        achievement_ids_to_block=CRAFTAX_CLASSIC_INTERMEDIATE_ACHIEVEMENT_IDS,
        remove_health_reward=True,
        episode_max_steps=2048,
        training_mode="baseline",
        selected_intrinsic_modules=("rnd",),
        baseline_fixed_training_alpha=(0.8, 0.2),
        encoder_mode="craftax_structured",
        num_envs_per_batch=1024,
        num_steps_per_env=8192,
        num_steps_per_update=512,
        eval_every_n_batches=1,
        eval_num_envs=512,
        eval_num_episodes=2,
        evaluation_alphas=((0.8, 0.2), (1.0, 0.0)),
        update_epochs=1,
        num_minibatches=16,
        lr=2e-4,
        clip_eps=0.2,
        gae_lambda=0.95,
        ent_coef=0.01,
        obs_emb_dim=128,
        past_context_length=64,
        subsequence_length_in_loss_calculation=32,
        num_attn_heads=4,
        num_transformer_blocks=1,
        transformer_hidden_states_dim=128,
        qkv_features=128,
        head_hidden_dim=128,
        enable_wandb=True,
        wandb_entity="openendedness-2026",
        is_timing_run=False,
    )


def build_run_config(*, smoke_run: bool) -> TrainConfig:
    """Return either the smoke or the full training preset."""
    if smoke_run:
        return build_smoke_run_config()
    return build_training_run_config()


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for the standalone training entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-run",
        action="store_true",
        help="Use the smoke-run config instead of the full training config.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Persist the checkpoint and metrics artifact after training finishes.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run training from the CLI using the selected preset."""
    args = parse_args(argv)
    config = build_run_config(smoke_run=args.smoke_run)
    run_main_algo_training(config=config, save_results=args.save_results)


if __name__ == "__main__":
    main()
