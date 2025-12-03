from datetime import datetime

project = "SNSpec"
author = "Cristine Koelln"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]
autosummary_generate = True
autodoc_typehints = "none"
autodoc_default_options = {"members": True, "undoc-members": True, "show-inheritance": True}
autodoc_mock_imports = ["astropy", "specutils", "numpy", "scipy", "pandas", "matplotlib", "scienceplots"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "titles_only": False,
}
html_static_path = ["_static"]

nitpick_ignore = [
    ('py:class', 'Path'),
]


# --- Automatic docstring cleaner --------------------------------------------
import re
from sphinx.ext.autodoc import between

def clean_docstring(doc: str) -> str:
    """Return a cleaned version of a docstring to avoid Sphinx warnings."""

    if not isinstance(doc, str):
        return doc

    # Remove undefined |resid| substitution
    doc = doc.replace("|resid|", "residuals")

    # Fix unbalanced * or ` markers
    if doc.count("*") % 2 == 1:
        doc = doc.replace("*", "")
    if doc.count("`") % 2 == 1:
        doc = doc.replace("`", "")

    # Remove gross indent errors
    lines = doc.splitlines()
    cleaned = []
    for ln in lines:
        if re.match(r"^\s{8,}\S", ln):
            cleaned.append(ln.lstrip())
        else:
            cleaned.append(ln)
    doc = "\n".join(cleaned)

    # Fix definition lists missing blank lines
    deflist_fixed = []
    lns = doc.splitlines()
    for i, ln in enumerate(lns):
        deflist_fixed.append(ln)
        if (
            i < len(lns) - 1
            and re.match(r"^\S", ln)
            and re.match(r"^\S", lns[i + 1])
        ):
            deflist_fixed.append("")
    doc = "\n".join(deflist_fixed)

    # Fix bare 'Path' references → avoid duplicate object description warnings
    doc = re.sub(
        r'\bPath\b',
        r':class:`~pathlib.Path`',
        doc
    )
    return doc  #Return a cleaned version of a docstring to avoid Sphinx warnings."""



def process_all_docstrings(app, what, name, obj, options, lines):
    """Hook called by autodoc for every docstring."""
    if not lines:
        return

    # Join → clean → split back
    cleaned = clean_docstring("\n".join(lines))
    lines[:] = cleaned.split("\n")


def setup(app):
    # Run our cleaner **after** autodoc retrieves docstrings
    app.connect("autodoc-process-docstring", process_all_docstrings)
    return {"version": "0.1", "parallel_read_safe": True}