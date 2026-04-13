# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add the project root so that autodoc can find the package.
sys.path.insert(0, os.path.abspath(".."))

from customhys import __author__, __version__

# -- Project information ------------------------------------------------------
project = "CUSTOMHyS"
copyright = "2025, Jorge Mario Cruz-Duarte"
author = __author__
version = __version__
release = __version__

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file parsers
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master document
master_doc = "index"

# -- MyST-Parser configuration ------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

# -- autodoc configuration ----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

# Mock heavy / optional imports so RTD builds succeed
autodoc_mock_imports = [
    "tensorflow",
    "tensorflow_macos",
    "tensorflow_metal",
    "tf",
]

# -- Napoleon configuration ---------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- intersphinx configuration ------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- HTML output options ------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}
html_static_path = ["_static"]
html_logo = "../docfiles/chm_logo.png"
html_favicon = "../docfiles/chm_logo.png"

# Suppress noisy docstring formatting warnings from legacy code
suppress_warnings = ["docutils"]

# -- LaTeX output options -----------------------------------------------------
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
}


# -- Custom CSS / JS ----------------------------------------------------------
def setup(app):
    app.add_css_file("custom.css")
    app.add_js_file("beta_banner.js")
