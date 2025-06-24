#!/usr/bin/env python3

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("_ext"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), "doc"))


# -- Project information -----------------------------------------------------

project = "Mr Mustard"
copyright = "2022, Xanadu Quantum Technologies"  # noqa: A001
author = "Filippo Miatto"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
import mrmustard as mm  # noqa: E402

release = mm.__version__

# The short X.Y version.
version = re.match(r"^(\d+\.\d+)", release).expand(r"\1")


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "edit_on_github",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx_automodapi.automodapi",
    "sphinx_copybutton",
    "m2r2",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

autosummary_generate = True
autosummary_imported_members = False
automodapi_toctreedirnm = "code/api"
automodsumm_inherited_members = True

mathjax_path = (
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
)

bibtex_bibfiles = ["references.bib"]


# -- Options for HTML output -------------------------------------------------

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    "**": [
        "searchbox.html",
        "globaltoc.html",
    ],
}

# Output file base name for HTML help builder.
htmlhelp_basename = "MrMustarddoc"

edit_on_github_project = "XanaduAI/MrMustard"
edit_on_github_branch = "master/doc"

# the order in which autodoc lists the documented members
autodoc_member_order = "bysource"

# inheritance_diagram graphviz attributes
inheritance_node_attrs = {"color": "lightskyblue1", "style": "filled"}


# -- Xanadu theme ---------------------------------------------------------
html_theme = "xanadu"

html_theme_options = {
    "navbar_name": "Mr Mustard",
    "navbar_logo_path": "_static/mm_logo.png",
    "navbar_right_links": [
        {
            "name": "GitHub",
            "href": "https://github.com/XanaduAI/MrMustard",
            "icon": "fab fa-github",
        },
    ],
    "extra_copyrights": [
        "TensorFlow, the TensorFlow logo, and any related marks are trademarks of Google Inc.",
    ],
    "google_analytics_tracking_id": "UA-116279123-2",
    "prev_next_button_colour": "#b79226",
    "prev_next_button_hover_colour": "#d7b348",
    "toc_marker_colour": "#b79226",
    "table_header_background_colour": "#ffdce5",
    "border_colour": "#b79226",
    "text_accent_colour": "#b79226",
}
