"""nohup poetry run python -m crew.experiments.fixed_weighting_runs >& nohup_2.out &"""

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
# `poetry run python -m crew.experiments.fixed_weighting_runs`
ENV_ID = "Craftax-Classic-Symbolic-v1"

# Keep these as Craftax Classic `Achievement` enum members.
EXTRINSIC_ACHIEVEMENTS = (
    Achievement.MAKE_IRON_PICKAXE,
    Achievement.PLACE_FURNACE,
)

# Choose from the registered intrinsic module names. If there are more than two,
# two are sampled once and reused for every fixed weighting and seed.
INTRINSIC_MODULE_CANDIDATES = ("rnd", "icm")
INTRINSIC_MODULE_SELECTION_SEED = 0

EVALUATION_GRID_SIZE = 8
TRAIN_SEEDS = (1, 2, 3)
SAVE_RESULTS = True

# This does not count the separate extrinsic-only run.
NUM_FIXED_WEIGHTINGS = 3
FIXED_WEIGHTING_SELECTION_SEED = 2222

# Set this from 0, 1, or 2 depending on the GPU you are running on.
WORKER_INDEX = 2
RUN_EXTRINSIC_ONLY_BASELINE = WORKER_INDEX == 0  # To avoid duplicates


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

    valid_grid_alphas = build_two_intrinsic_evaluation_alphas(EVALUATION_GRID_SIZE)
    extrinsic_only_alpha = (1.0, 0.0, 0.0)
    candidate_fixed_alphas = tuple(alpha for alpha in valid_grid_alphas if alpha != extrinsic_only_alpha)
    if NUM_FIXED_WEIGHTINGS > len(candidate_fixed_alphas):
        raise ValueError("NUM_FIXED_WEIGHTINGS is larger than the number of available fixed weightings in the grid.")
    if WORKER_INDEX * NUM_FIXED_WEIGHTINGS >= len(candidate_fixed_alphas):
        raise ValueError("WORKER_INDEX is too large, it exceeds the available fixed weightings.")

    # Shuffle all available alphas with a common seed, then take a non-overlapping slice for this worker
    shuffled_candidate_alphas = random.Random(FIXED_WEIGHTING_SELECTION_SEED).sample(
        list(candidate_fixed_alphas), k=len(candidate_fixed_alphas)
    )
    start_idx = WORKER_INDEX * NUM_FIXED_WEIGHTINGS
    end_idx = start_idx + NUM_FIXED_WEIGHTINGS
    sampled_fixed_alphas = tuple(shuffled_candidate_alphas[start_idx:end_idx])

    total_achievements = len(ORDERED_ACHIEVEMENTS_BY_ENV[ENV_ID])
    num_extrinsic_achievements = total_achievements - len(achievement_ids_to_block)
    total_runs = (1 + len(sampled_fixed_alphas)) * len(TRAIN_SEEDS)

    print(
        f"extrinsic_achievements={tuple(achievement.name.lower() for achievement in EXTRINSIC_ACHIEVEMENTS)} "
        f"({num_extrinsic_achievements}/{total_achievements})"
    )
    print(f"selected_intrinsic_modules={selected_intrinsic_modules}")
    print(f"extrinsic_only_alpha={extrinsic_only_alpha}")
    print(f"sampled_fixed_alphas={sampled_fixed_alphas}")
    print(f"train_seeds={tuple(TRAIN_SEEDS)}")
    print(f"total_runs={total_runs}")

    alphas = (extrinsic_only_alpha, *sampled_fixed_alphas) if RUN_EXTRINSIC_ONLY_BASELINE else sampled_fixed_alphas
    for fixed_alpha in alphas:
        for train_seed in TRAIN_SEEDS:
            config = TrainConfig(
                training_mode="baseline",
                env_id=ENV_ID,
                achievement_ids_to_block=achievement_ids_to_block,
                train_seed=train_seed,
                fixed_reset_seed=train_seed + 100,
                selected_intrinsic_modules=selected_intrinsic_modules,
                baseline_fixed_training_alpha=fixed_alpha,
                eval_every_n_batches=2,
                eval_num_envs=256,
                eval_num_episodes=1,
                total_timesteps=1_000_000_000,
                video_num_episodes=2,
            )
            print(f"Starting fixed weighting run for alpha={fixed_alpha} train_seed={train_seed}")
            run_main_algo_training(config=config, save_results=SAVE_RESULTS)


if __name__ == "__main__":
    main()
