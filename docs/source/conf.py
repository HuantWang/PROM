# Configuration file for the Sphinx documentation builder.
# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../examples/case_study'))
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../examples/case_study/BugD'))
sys.path.insert(0, os.path.abspath('../../examples/case_study/DeviceM'))
sys.path.insert(0, os.path.abspath('../../examples/case_study/Loop'))
sys.path.insert(0, os.path.abspath('../../examples/case_study/Thread'))
sys.path.insert(0, os.path.abspath('../../examples/case_study/tlp/scripts'))
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Prom'
copyright = '2024, _'
author = '_'
release = 'v0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx.ext.autosectionlabel",
    "sphinxemoji.sphinxemoji",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"