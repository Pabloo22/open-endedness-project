"""Small helpers shared by the paper experiment scripts."""

from __future__ import annotations

from collections.abc import Sequence

from curemix.experiments.identity import ORDERED_ACHIEVEMENTS_BY_ENV


def build_achievement_ids_to_block(env_id: str, extrinsic_achievements: Sequence[object]) -> tuple[int, ...]:
    extrinsic_ids = {int(getattr(achievement, "value", achievement)) for achievement in extrinsic_achievements}
    return tuple(
        int(achievement.value)
        for achievement in ORDERED_ACHIEVEMENTS_BY_ENV[env_id]
        if int(achievement.value) not in extrinsic_ids
    )


def build_two_intrinsic_evaluation_alphas(grid_size: int) -> tuple[tuple[float, float, float], ...]:
    evaluation_alphas: list[tuple[float, float, float]] = []
    for total_intrinsic_units in range(grid_size):
        extrinsic_units = grid_size - total_intrinsic_units
        for first_intrinsic_units in range(total_intrinsic_units, -1, -1):
            second_intrinsic_units = total_intrinsic_units - first_intrinsic_units
            evaluation_alphas.append(
                (
                    extrinsic_units / grid_size,
                    first_intrinsic_units / grid_size,
                    second_intrinsic_units / grid_size,
                )
            )
    return tuple(evaluation_alphas)
