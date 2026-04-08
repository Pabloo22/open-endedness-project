import unittest
from types import SimpleNamespace

from curemix.main_algo.main_loop import _validate_supported_curriculum_intrinsic_modules


class TestCurriculumIntrinsicModuleValidation(unittest.TestCase):
    def test_curriculum_rejects_unsupported_intrinsic_modules(self):
        with self.assertRaisesRegex(ValueError, "only supports delayed-sync intrinsic modules"):
            _validate_supported_curriculum_intrinsic_modules(
                intrinsic_modules=(SimpleNamespace(name="ngu"),),
            )


if __name__ == "__main__":
    unittest.main()
