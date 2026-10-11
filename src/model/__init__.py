"""Re-exports GestureClassifier, Classifier.

Owner: Siri (Classifier).
Specification: docs/interfaces.md §4.5.

Imports are lazy so `import src.model` does not pull in scikit-learn until a name is used.
"""

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "Classifier": "src.model.predict",
    "GestureClassifier": "src.model.predict",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import a re-exported name from its submodule on first use."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.model' has no attribute {name!r}")
    return getattr(import_module(_EXPORTS[name]), name)
