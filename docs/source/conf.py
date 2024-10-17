# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

# Enable access to the PICOS module.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qics._version import __version__  # noqa isort:skip

project = "QICS"
copyright = "2024, Kerry He, James Saunderson, and Hamza Fawzi"
author = "Kerry He, James Saunderson, and Hamza Fawzi"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_tabs.tabs",
    "numpydoc",
]

numpydoc_show_class_members = False

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3

# Configure intersphinx.
intersphinx_cache_limit = 10
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "picos": ("https://picos-api.gitlab.io/picos/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# sphinx-copybutton configurations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/kerry-he/qics",
    "accent_color": "grass",
}
