from __future__ import annotations

import random

from craftax.craftax_classic.constants import Achievement

from crew.experiments.identity import ORDERED_ACHIEVEMENTS_BY_ENV
from crew.experiments.paper_run_utils import (
    build_achievement_ids_to_block,
    build_two_intrinsic_evaluation_alphas,
)
from crew.experiments.run_training import run_main_algo_training
from crew.main_algo.config import TrainConfig

# Edit these variables directly before running:
# `poetry run python -m crew.experiments.curriculum_runs`
ENV_ID = "Craftax-Classic-Symbolic-v1"

# Keep these as Craftax Classic `Achievement` enum members.
EXTRINSIC_ACHIEVEMENTS = (
    Achievement.PLACE_FURNACE,
    Achievement.MAKE_IRON_PICKAXE,
)

# Choose from the registered intrinsic module names. Two are sampled once and reused for every seed.
INTRINSIC_MODULE_CANDIDATES = ("rnd", "icm")
INTRINSIC_MODULE_SELECTION_SEED = 0

EVALUATION_GRID_SIZE = 8

TRAIN_SEEDS = (1, 2, 3)
SAVE_RESULTS = True


def main() -> None:
    achievement_ids_to_block = build_achievement_ids_to_block(ENV_ID, EXTRINSIC_ACHIEVEMENTS)
    if len(INTRINSIC_MODULE_CANDIDATES) < 2:
        raise ValueError("INTRINSIC_MODULE_CANDIDATES must contain at least two modules.")

    if len(INTRINSIC_MODULE_CANDIDATES) > 2:
        selected_intrinsic_modules = tuple(
            sorted(random.Random(INTRINSIC_MODULE_SELECTION_SEED).sample(list(INTRINSIC_MODULE_CANDIDATES), k=2))
        )
    else:
        selected_intrinsic_modules = tuple(sorted(INTRINSIC_MODULE_CANDIDATES))

    evaluation_alphas = build_two_intrinsic_evaluation_alphas(EVALUATION_GRID_SIZE)
    total_achievements = len(ORDERED_ACHIEVEMENTS_BY_ENV[ENV_ID])
    num_extrinsic_achievements = total_achievements - len(achievement_ids_to_block)

    print(
        f"extrinsic_achievements={tuple(achievement.name.lower() for achievement in EXTRINSIC_ACHIEVEMENTS)} "
        f"({num_extrinsic_achievements}/{total_achievements})"
    )
    print(f"selected_intrinsic_modules={selected_intrinsic_modules}")
    print(f"num_evaluation_alphas={len(evaluation_alphas)}")
    print(f"train_seeds={tuple(TRAIN_SEEDS)}")

    for train_seed in TRAIN_SEEDS:
        config = TrainConfig(
            training_mode="curriculum",
            env_id=ENV_ID,
            achievement_ids_to_block=achievement_ids_to_block,
            train_seed=train_seed,
            fixed_reset_seed=train_seed + 100,
            selected_intrinsic_modules=selected_intrinsic_modules,
            evaluation_alphas=evaluation_alphas,
            eval_every_n_batches=5,
            eval_num_envs=32,
            eval_num_episodes=8,
            total_timesteps=1_000_000_000,
        )
        print(f"Starting curriculum run for train_seed={train_seed}")
        run_main_algo_training(config=config, save_results=SAVE_RESULTS)


if __name__ == "__main__":
    main()
