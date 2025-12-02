# src/single_sne/__init__.py

"""
Top-level package for SNSpec.

This lazily re-exports everything from .snspec so that users can do:

    import single_sne
    single_sne.some_function(...)

without us having to manually mirror every symbol here.
"""

import importlib


def _load_snspec():
    """Helper to import the core snspec module."""
    return importlib.import_module(".snspec", __name__)


def __getattr__(name):
    snspec = _load_snspec()
    return getattr(snspec, name)


def __dir__():
    snspec = _load_snspec()
    return sorted(set(globals().keys()) | set(dir(snspec)))


try:
    _snspec = _load_snspec()
    __all__ = getattr(
        _snspec,
        "__all__",
        [n for n in dir(_snspec) if not n.startswith("_")],
    )
except Exception:
    # If snspec fails to import (e.g. during an incomplete editable install),
    # keep the top-level namespace minimal instead of crashing.
    __all__ = []