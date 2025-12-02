# Lazy re-export of everything from .snspec
# - No need to update this file when you add new functions.
# - Keeps import fast and the top-level namespace clean.

import importlib

def __getattr__(name):
    snspec = importlib.import_module(".single_sne", __name__)
    return getattr(snspec, name)

def __dir__():
    snspec = importlib.import_module(".single_sne", __name__)
    return sorted(set(globals().keys()) | set(dir(snspec)))

# Mirror __all__ from snspec (if defined) so that help(single_sne) looks nice.
try:
    _snspec = importlib.import_module(".single_sne", __name__)
    __all__ = getattr(_snspec, "__all__", [n for n in dir(_snspec) if not n.startswith("_")])
except Exception:
    # During editable install, snspec might fail temporarily; keep namespace minimal.
    __all__ = []
