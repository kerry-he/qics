# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

project = "QICS"
copyright = "2024, Kerry He, James Saunderson, and Hamza Fawzi"
author = "Kerry He, James Saunderson, and Hamza Fawzi"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
]

autodoc_mock_imports = ["numpy", "scipy", "numba"]
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/kerry-he/qics",
    "accent_color": "grass",
}
