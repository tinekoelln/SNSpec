"""
Initialize the spectra subpackage.
We do NOT import submodules here to avoid circular import problems.
"""

__all__ = [
    "spectra",
    "process_epochs",
    "stitching",
    "instruments",
    "units",
    "xsh_merge",
]

from importlib import import_module

def __getattr__(name):
    if name in __all__:
        return import_module(f".{name}", __name__)
    raise AttributeError(name)