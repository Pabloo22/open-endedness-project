import unittest
from types import SimpleNamespace

from craftax.craftax_classic.constants import Achievement

from curemix.experiments.identity import (
    build_experiment_identity,
    build_intrinsic_rewards_used,
    build_task_identifier,
)


def _blocked_ids_for_active_achievements(*active_achievements: Achievement) -> tuple[int, ...]:
    active_ids = {achievement.value for achievement in active_achievements}
    return tuple(
        achievement.value
        for achievement in Achievement
        if achievement.value not in active_ids
    )


class TestExperimentIdentity(unittest.TestCase):
    def test_build_task_identifier_uses_readable_active_achievement_slugs(self):
        task_identifier = build_task_identifier(
            env_id="Craftax-Classic-Symbolic-v1",
            achievement_ids_to_block=_blocked_ids_for_active_achievements(
                Achievement.COLLECT_WOOD,
                Achievement.PLACE_TABLE,
            ),
        )

        self.assertEqual(task_identifier, "collect_wood+place_table")

    def test_build_task_identifier_uses_all_achievements_when_nothing_is_blocked(self):
        self.assertEqual(
            build_task_identifier(
                env_id="Craftax-Classic-Symbolic-v1",
                achievement_ids_to_block=(),
            ),
            "all_achievements",
        )

    def test_build_intrinsic_rewards_used_formats_baseline_weights(self):
        self.assertEqual(
            build_intrinsic_rewards_used(
                training_mode="baseline",
                selected_intrinsic_modules=("icm", "rnd"),
                baseline_fixed_training_alpha=(0.7, 0.1, 0.2),
            ),
            "icm0p1+rnd0p2",
        )

    def test_build_experiment_identity_builds_consistent_path_and_wandb_components(self):
        config = SimpleNamespace(
            training_mode="baseline",
            env_id="Craftax-Classic-Symbolic-v1",
            achievement_ids_to_block=_blocked_ids_for_active_achievements(
                Achievement.COLLECT_IRON,
                Achievement.MAKE_WOOD_PICKAXE,
            ),
            selected_intrinsic_modules=("rnd",),
            baseline_fixed_training_alpha=(0.8, 0.2),
            train_seed=3,
        )

        identity = build_experiment_identity(config)

        self.assertEqual(identity.task_identifier, "make_wood_pickaxe+collect_iron")
        self.assertEqual(identity.algorithm_id, "baseline")
        self.assertEqual(identity.intrinsic_rewards_used, "rnd0p2")
        self.assertEqual(identity.run_group, "make_wood_pickaxe+collect_iron/baseline/rnd0p2")
        self.assertEqual(identity.run_name, "make_wood_pickaxe+collect_iron/baseline/rnd0p2|seed3")
        self.assertEqual(
            identity.tags,
            (
                "task:make_wood_pickaxe+collect_iron",
                "algo:baseline",
                "intr:rnd0p2",
            ),
        )


if __name__ == "__main__":
    unittest.main()
