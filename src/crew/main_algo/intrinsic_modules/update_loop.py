from typing import Any

import jax

from crew.main_algo.intrinsic_modules.api import IntrinsicModule
from crew.main_algo.types import (
    IntrinsicModulesUpdateData,
    IntrinsicStates,
)


def update_intrinsic_modules(
    rng: jax.Array,
    intrinsic_modules: tuple[IntrinsicModule, ...],
    intrinsic_states: IntrinsicStates,
    intrinsic_modules_update_data: IntrinsicModulesUpdateData,
    config: Any,
) -> tuple[jax.Array, IntrinsicStates, dict[str, jax.Array]]:
    """Update each intrinsic module once using data from the fixed-alpha window."""
    updated_states = []
    metrics: dict[str, jax.Array] = {}
    for module, module_state in zip(intrinsic_modules, intrinsic_states, strict=True):
        rng, module_rng = jax.random.split(rng)
        updated_module_state, module_metrics = module.update(
            rng=module_rng,
            module_state=module_state,
            transitions=intrinsic_modules_update_data,
            config=config,
        )
        updated_states.append(updated_module_state)
        metrics = metrics | module_metrics
    return rng, tuple(updated_states), metrics
