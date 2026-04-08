from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from craftax.craftax.constants import Achievement as CraftaxAchievement
from craftax.craftax_classic.constants import Achievement as CraftaxClassicAchievement


ORDERED_ACHIEVEMENTS_BY_ENV = {
    "Craftax-Classic-Symbolic-v1": tuple(
        sorted(CraftaxClassicAchievement, key=lambda achievement: int(achievement.value))
    ),
    "Craftax-Symbolic-v1": tuple(sorted(CraftaxAchievement, key=lambda achievement: int(achievement.value))),
}


@dataclass(frozen=True)
class ExperimentIdentity:
    task_identifier: str
    algorithm_id: str
    intrinsic_rewards_used: str
    run_group: str
    run_name: str
    tags: tuple[str, ...]


def build_task_identifier(env_id: str, achievement_ids_to_block: Sequence[int]) -> str:
    ordered_achievements = ORDERED_ACHIEVEMENTS_BY_ENV[env_id]
    blocked_ids = {int(achievement_id) for achievement_id in achievement_ids_to_block}
    if not blocked_ids:
        return "all_achievements"

    return "+".join(
        achievement.name.lower()
        for achievement in ordered_achievements
        if int(achievement.value) not in blocked_ids
    )


def build_intrinsic_rewards_used(
    training_mode: str,
    selected_intrinsic_modules: Sequence[str] | None,
    baseline_fixed_training_alpha: Sequence[float] | None,
) -> str:
    modules = tuple(selected_intrinsic_modules or ())
    if training_mode == "curriculum":
        return "+".join(modules) if modules else "none"
    if not modules:
        return "none"

    if baseline_fixed_training_alpha is None:
        intrinsic_weights = (0.0,) * len(modules)
    else:
        intrinsic_weights = tuple(float(weight) for weight in baseline_fixed_training_alpha[1:])

    weighted_modules = []
    for module_name, module_weight in zip(modules, intrinsic_weights, strict=True):
        weight_text = f"{float(module_weight):.12f}".rstrip("0").rstrip(".")
        if weight_text in ("", "-0"):
            weight_text = "0"
        weighted_modules.append(f"{module_name}{weight_text.replace('.', 'p')}")
    return "+".join(weighted_modules)


def build_experiment_identity(config: Any) -> ExperimentIdentity:
    algorithm_id = config.training_mode
    task_identifier = build_task_identifier(
        env_id=config.env_id,
        achievement_ids_to_block=config.achievement_ids_to_block,
    )
    intrinsic_rewards_used = build_intrinsic_rewards_used(
        training_mode=algorithm_id,
        selected_intrinsic_modules=config.selected_intrinsic_modules,
        baseline_fixed_training_alpha=getattr(config, "baseline_fixed_training_alpha", None),
    )
    run_group = f"{task_identifier}/{algorithm_id}/{intrinsic_rewards_used}"
    run_name = f"{run_group}|seed{config.train_seed}"
    tags = (
        f"task:{task_identifier}",
        f"algo:{algorithm_id}",
        f"intr:{intrinsic_rewards_used}",
    )
    return ExperimentIdentity(
        task_identifier=task_identifier,
        algorithm_id=algorithm_id,
        intrinsic_rewards_used=intrinsic_rewards_used,
        run_group=run_group,
        run_name=run_name,
        tags=tags,
    )
