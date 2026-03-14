"""Registry for intrinsic reward modules."""

from crew.main_algo.intrinsic_modules.api import IntrinsicModule
from crew.main_algo.intrinsic_modules.rnd import RNDIntrinsicModule
from crew.main_algo.intrinsic_modules.icm import ICMIntrinsicModule

_REGISTRY: dict[str, IntrinsicModule] = {
    "rnd": RNDIntrinsicModule(),
    "icm": ICMIntrinsicModule(),
}


def get_registered_intrinsic_module_names() -> tuple[str, ...]:
    """Return all registered module names sorted for stable metrics keys."""
    return tuple(sorted(_REGISTRY))


def get_intrinsic_module(name: str) -> IntrinsicModule:
    """Resolve one intrinsic module by name."""
    if name not in _REGISTRY:
        msg = f"Unknown intrinsic module {name!r}. " f"Available modules: {get_registered_intrinsic_module_names()}."
        raise ValueError(msg)
    return _REGISTRY[name]


def get_selected_intrinsic_modules(
    module_names: tuple[str, ...],
) -> tuple[IntrinsicModule, ...]:
    """Resolve a static tuple of selected intrinsic modules."""
    return tuple(get_intrinsic_module(name) for name in module_names)
