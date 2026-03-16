import copy
from typing import Any

from crew.experiments.tuning_configs import (
    get_lightweight_base_params,
    get_generic_search_space_v1,
    get_rnd_base_config_v1,
    get_rnd_search_space_v1,
    get_icm_base_config_v1,
    get_icm_search_space_v1,
    get_ngu_base_config_v1,
    get_ngu_search_space_v1,
    get_curriculum_search_space_v1,
)


ACTIVE_GENERIC_BASE_CONFIG = get_lightweight_base_params()
ACTIVE_GENERIC_SEARCH_SPACE = get_generic_search_space_v1()

ACTIVE_RND_BASE_CONFIG = get_rnd_base_config_v1()
ACTIVE_ICM_BASE_CONFIG = get_icm_base_config_v1()
ACTIVE_NGU_BASE_CONFIG = get_ngu_base_config_v1()

ACTIVE_RND_SEARCH_SPACE = get_rnd_search_space_v1()
ACTIVE_ICM_SEARCH_SPACE = get_icm_search_space_v1()
ACTIVE_NGU_SEARCH_SPACE = get_ngu_search_space_v1()


def _build_curriculum_placeholder_base_config(module_names: tuple[str, ...]) -> dict[str, Any]:
    num_rewards = 1 + len(module_names)
    equal_intrinsic_weight = 1.0 / num_rewards
    extrinsic_weight = 1.0 - (equal_intrinsic_weight * len(module_names))
    mixed_alpha = (extrinsic_weight, *(equal_intrinsic_weight for _ in module_names))
    extrinsic_only_alpha = tuple(1.0 if idx == 0 else 0.0 for idx in range(num_rewards))
    return {
        "training_mode": "curriculum",
        "selected_intrinsic_modules": module_names,
        "evaluation_alphas": (mixed_alpha, extrinsic_only_alpha),
    }


ACTIVE_INTRINSIC_BASE_CONFIGS = {
    "rnd": {
        **ACTIVE_GENERIC_BASE_CONFIG,
        **ACTIVE_RND_BASE_CONFIG,
        "training_mode": "baseline",
    },
    "icm": {
        **ACTIVE_GENERIC_BASE_CONFIG,
        **ACTIVE_ICM_BASE_CONFIG,
        "training_mode": "baseline",
    },
    "ngu": {
        **ACTIVE_GENERIC_BASE_CONFIG,
        **ACTIVE_NGU_BASE_CONFIG,
        "training_mode": "baseline",
    },
}
ACTIVE_INTRINSIC_SEARCH_SPACES = {
    "rnd": ACTIVE_RND_SEARCH_SPACE,
    "icm": ACTIVE_ICM_SEARCH_SPACE,
    "ngu": ACTIVE_NGU_SEARCH_SPACE,
}

ACTIVE_CURRICULUM_BASE_CONFIGS: dict[tuple[str, ...], dict[str, Any]] = {
    ("rnd",): _build_curriculum_placeholder_base_config(("rnd",)),
    ("icm",): _build_curriculum_placeholder_base_config(("icm",)),
    ("ngu",): _build_curriculum_placeholder_base_config(("ngu",)),
    ("rnd", "icm"): _build_curriculum_placeholder_base_config(("rnd", "icm")),
    ("rnd", "ngu"): _build_curriculum_placeholder_base_config(("rnd", "ngu")),
    ("icm", "ngu"): _build_curriculum_placeholder_base_config(("icm", "ngu")),
    ("rnd", "icm", "ngu"): _build_curriculum_placeholder_base_config(("rnd", "icm", "ngu")),
}
ACTIVE_CURRICULUM_SEARCH_SPACE = get_curriculum_search_space_v1()


def get_generic_base_config() -> dict[str, Any]:
    """Return the current generic-phase base config."""
    return copy.deepcopy(ACTIVE_GENERIC_BASE_CONFIG)


def get_generic_search_space() -> dict[str, Any]:
    """Return the current generic-phase search space."""
    return copy.deepcopy(ACTIVE_GENERIC_SEARCH_SPACE)


def get_intrinsic_base_config(intrinsic_module: str) -> dict[str, Any]:
    """Return the current intrinsic-phase base config for one module."""
    if intrinsic_module not in ACTIVE_INTRINSIC_BASE_CONFIGS:
        msg = (
            f"Unsupported intrinsic module {intrinsic_module!r} for intrinsic tuning presets. "
            f"Supported modules: {tuple(ACTIVE_INTRINSIC_BASE_CONFIGS)}."
        )
        raise ValueError(msg)
    return copy.deepcopy(ACTIVE_INTRINSIC_BASE_CONFIGS[intrinsic_module])


def get_intrinsic_search_space(intrinsic_module: str) -> dict[str, Any]:
    """Return the current intrinsic-phase search space for one module."""
    if intrinsic_module not in ACTIVE_INTRINSIC_SEARCH_SPACES:
        msg = (
            f"Unsupported intrinsic module {intrinsic_module!r} for intrinsic search spaces. "
            f"Supported modules: {tuple(ACTIVE_INTRINSIC_SEARCH_SPACES)}."
        )
        raise ValueError(msg)
    return copy.deepcopy(ACTIVE_INTRINSIC_SEARCH_SPACES[intrinsic_module])


def get_curriculum_base_config_for_modules(
    module_names: tuple[str, ...],
) -> dict[str, Any]:
    """Return the current curriculum-phase base config for one intrinsic-module set."""
    normalized_module_names = normalize_intrinsic_modules(module_names)
    normalized_presets = {normalize_intrinsic_modules(k): v for k, v in ACTIVE_CURRICULUM_BASE_CONFIGS.items()}
    if normalized_module_names not in normalized_presets:
        msg = (
            f"Unsupported intrinsic modules {normalized_module_names!r} for curriculum tuning presets. "
            f"Supported modules: {tuple(normalized_presets)}."
        )
        raise ValueError(msg)
    return copy.deepcopy(normalized_presets[normalized_module_names])


def normalize_intrinsic_modules(module_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return a stable intrinsic-module tuple for preset lookups."""
    if not module_names:
        msg = "At least one intrinsic module must be provided."
        raise ValueError(msg)
    return tuple(sorted(module_names))


def get_curriculum_search_space() -> dict[str, Any]:
    """Return the current curriculum-phase search space."""
    return copy.deepcopy(ACTIVE_CURRICULUM_SEARCH_SPACE)
