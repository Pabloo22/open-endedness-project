import copy
from typing import Any

from crew.experiments.tuning_configs import (
    get_lightweight_base_params,
    get_generic_search_space_v1,
    get_rnd_base_config_v1,
    get_rnd_search_space_v1,
    get_curriculum_search_space_v1,
)


ACTIVE_GENERIC_BASE_CONFIG = get_lightweight_base_params()
ACTIVE_GENERIC_SEARCH_SPACE = get_generic_search_space_v1()

ACTIVE_RND_BASE_CONFIG = get_rnd_base_config_v1()
ACTIVE_ICM_BASE_CONFIG: dict[str, Any] = {}  # TODO: Add ICM base config
ACTIVE_NGU_BASE_CONFIG: dict[str, Any] = {}  # TODO: Add NGU base config

ACTIVE_INTRINSIC_BASE_CONFIGS = {
    "rnd": {
        **ACTIVE_GENERIC_BASE_CONFIG,
        **ACTIVE_RND_BASE_CONFIG,
        "training_mode": "intrinsic",
    }
}
ACTIVE_INTRINSIC_SEARCH_SPACES = {
    "rnd": get_rnd_search_space_v1(),
}

ACTIVE_CURRICULUM_BASE_CONFIGS: dict[tuple[str, ...], dict[str, Any]] = {
    ("rnd", "icm", "ngu"): {
        **ACTIVE_GENERIC_BASE_CONFIG,
        **ACTIVE_RND_BASE_CONFIG,
        **ACTIVE_ICM_BASE_CONFIG,
        **ACTIVE_NGU_BASE_CONFIG,
        "training_mode": "curriculum",
    }
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
    print("-" * 20)
    print(normalized_presets)
    print("-" * 20)
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
