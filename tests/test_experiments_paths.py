import unittest
from pathlib import Path
from types import SimpleNamespace

from craftax.craftax_classic.constants import Achievement

from crew.experiments.paths import build_best_weights_rollouts_path, build_trained_weights_path


class TestExperimentPaths(unittest.TestCase):
    def test_build_trained_weights_path_formats_intrinsic_module_combinations(self):
        blocked_ids = tuple(
            achievement.value
            for achievement in Achievement
            if achievement not in (Achievement.COLLECT_WOOD, Achievement.PLACE_TABLE)
        )
        config = SimpleNamespace(
            training_mode="curriculum",
            env_id="Craftax-Classic-Symbolic-v1",
            achievement_ids_to_block=blocked_ids,
            selected_intrinsic_modules=("icm", "ngu", "rnd"),
            baseline_fixed_training_alpha=None,
            train_seed=7,
            artifacts_root="/tmp/artifacts",
        )
        path = build_trained_weights_path(config=config)

        expected = (
            Path("/tmp/artifacts").resolve()
            / "training_results/collect_wood+place_table/curriculum/icm+ngu+rnd/seed7"
        )
        self.assertEqual(path, expected)

    def test_build_paths_use_none_when_intrinsic_modules_are_empty(self):
        training_config = SimpleNamespace(
            training_mode="baseline",
            env_id="Craftax-Classic-Symbolic-v1",
            achievement_ids_to_block=(),
            selected_intrinsic_modules=(),
            baseline_fixed_training_alpha=(1.0,),
            train_seed=11,
            artifacts_root="/tmp/artifacts",
        )
        rollouts_config = SimpleNamespace(
            training_mode="baseline",
            env_id="Craftax-Classic-Symbolic-v1",
            achievement_ids_to_block=(),
            selected_intrinsic_modules=(),
            baseline_fixed_training_alpha=(1.0,),
            train_seed=11,
            artifacts_root="/tmp/artifacts",
        )
        training_path = build_trained_weights_path(config=training_config)
        rollouts_path = build_best_weights_rollouts_path(config=rollouts_config)

        root = Path("/tmp/artifacts").resolve()
        self.assertEqual(
            training_path,
            root / "training_results/all_achievements/baseline/none/seed11",
        )
        self.assertEqual(
            rollouts_path,
            root / "best_weights_rollouts/all_achievements/baseline/none/seed11",
        )

    def test_build_paths_include_baseline_intrinsic_weights(self):
        config = SimpleNamespace(
            training_mode="baseline",
            env_id="Craftax-Classic-Symbolic-v1",
            achievement_ids_to_block=(),
            selected_intrinsic_modules=("rnd",),
            baseline_fixed_training_alpha=(0.8, 0.2),
            train_seed=5,
            artifacts_root="/tmp/artifacts",
        )

        path = build_trained_weights_path(config=config)

        self.assertEqual(
            path,
            Path("/tmp/artifacts").resolve() / "training_results/all_achievements/baseline/rnd0p2/seed5",
        )


if __name__ == "__main__":
    unittest.main()
