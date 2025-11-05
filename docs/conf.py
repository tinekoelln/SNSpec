from datetime import datetime

project = "SNSpec"
author = "Cristine Koelln"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.0.0"

extensions = []  # keep empty for first build; add autodoc later
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "furo"
html_static_path = ["_static"]
# If you later enable autodoc, you can also add:
# extensions += ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
# autosummary_generate = True
# autodoc_typehints = "description"
# autodoc_default_options = {"members": True, "undoc-members": True, "show-inheritance": True}
# autodoc_mock_imports = ["astropy", "specutils", "numpy", "scipy", "pandas", "matplotlib", "scienceplots"]
