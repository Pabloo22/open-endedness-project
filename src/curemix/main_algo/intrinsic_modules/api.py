"""Protocol definitions for intrinsic reward modules."""

from typing import Any, Protocol, runtime_checkable

import jax

from curemix.main_algo.types import (
    IntrinsicModuleState,
    IntrinsicModulesUpdateData,
    IntrinsicUpdateMetrics,
    TransitionDataBase,
)


@runtime_checkable
class IntrinsicModule(Protocol):
    """Common API that all intrinsic modules must satisfy."""

    name: str
    is_episodic: bool

    def init_state(
        self,
        rng: jax.Array,
        obs_shape: tuple[int, ...],
        config: Any,
    ) -> IntrinsicModuleState:
        """Initialize module state."""

    def compute_rewards(
        self,
        rng: jax.Array,
        module_state: IntrinsicModuleState,
        transitions: TransitionDataBase,
        config: Any,
    ) -> jax.Array:
        """Compute intrinsic rewards for the given transitions."""

    def update(
        self,
        rng: jax.Array,
        module_state: IntrinsicModuleState,
        transitions: IntrinsicModulesUpdateData,
        config: Any,
    ) -> tuple[IntrinsicModuleState, IntrinsicUpdateMetrics]:
        """Update module state and return metrics."""

    def done_mask(self, env_done: jax.Array, config: Any) -> jax.Array:
        """Compute the done mask."""
