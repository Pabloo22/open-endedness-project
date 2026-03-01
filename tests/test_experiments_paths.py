import unittest
from types import SimpleNamespace
from pathlib import Path

from crew.experiments.paths import build_best_weights_rollouts_path, build_trained_weights_path


class TestExperimentPaths(unittest.TestCase):
    def test_build_trained_weights_path_formats_intrinsic_module_combinations(self):
        config = SimpleNamespace(
            training_mode="curriculum",
            env_id="Craftax-Classic-Symbolic-v1",
            selected_intrinsic_modules=("rnd", "icm"),
            train_seed=7,
            artifacts_root="/tmp/artifacts",
        )
        path = build_trained_weights_path(config=config)

        expected = (
            Path("/tmp/artifacts").resolve()
            / "Craftax-Classic-Symbolic-v1/training_results/curriculum/rnd+icm/seed7"
        )
        self.assertEqual(path, expected)

    def test_build_paths_use_none_when_intrinsic_modules_are_empty(self):
        training_config = SimpleNamespace(
            training_mode="baseline",
            env_id="Craftax-Classic-Symbolic-v1",
            selected_intrinsic_modules=(),
            train_seed=11,
            artifacts_root="/tmp/artifacts",
        )
        rollouts_config = SimpleNamespace(
            training_mode="baseline",
            env_id="Craftax-Classic-Symbolic-v1",
            selected_intrinsic_modules=None,
            train_seed=11,
            artifacts_root="/tmp/artifacts",
        )
        training_path = build_trained_weights_path(config=training_config)
        rollouts_path = build_best_weights_rollouts_path(config=rollouts_config)

        root = Path("/tmp/artifacts").resolve()
        self.assertEqual(
            training_path,
            root / "Craftax-Classic-Symbolic-v1/training_results/baseline/none/seed11",
        )
        self.assertEqual(
            rollouts_path,
            root / "Craftax-Classic-Symbolic-v1/best_weights_rollouts/baseline/none/seed11",
        )


if __name__ == "__main__":
    unittest.main()
