project = "Open MatSciML Toolkit"
copyright = "2024, Intel Corporation"
author = "Intel Corporation"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]

# configure napoleon docstrings
napoleon_numpy_docstring = True
napoleon_preprocess_types = True

# ensure __init__ is used for class documentation
autoclass_content = "both"
