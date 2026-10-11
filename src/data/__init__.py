"""Re-exports Dataset, load_dataset, Splits, load_splits, load_split.

Owner: Shobita (Dataset).
Specification: docs/interfaces.md §4.5.

"""

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "Dataset": "src.data.dataset",
    "load_dataset": "src.data.dataset",
    "Splits": "src.data.split",
    "load_splits": "src.data.split",
    "load_split": "src.data.split",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import a re-exported name from its submodule on first use."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.data' has no attribute {name!r}")
    return getattr(import_module(_EXPORTS[name]), name)
