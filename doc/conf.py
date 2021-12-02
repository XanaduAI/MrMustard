#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import os, sys, re

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("_ext"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), "doc"))


# -- Project information -----------------------------------------------------

project = "Mr. Mustard"
copyright = "Copyright 2021, Xanadu Quantum Technologies Inc."
author = "Filippo Miatto"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
import mrmustard as mm

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
templates_path = ["_templates", "xanadu_theme"]

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

mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"

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
        "logo-text.html",
        "searchbox.html",
        "globaltoc.html",
        # 'sourcelink.html'
    ]
}

# Output file base name for HTML help builder.
htmlhelp_basename = "mrmustarddoc"

# -- Xanadu theme ---------------------------------------------------------
html_theme = "xanadu_theme"
html_theme_path = ["."]

# xanadu theme options (see theme.conf for more information)
html_theme_options = {
    # Set the path to a special layout to include for the homepage
    # "homepage": "special_index.html",
    # Set the name of the project to appear in the left sidebar.
    "project_nav_name": "Mr. Mustard",
    "touch_icon": "_static/logo_new.png",
    # Set GA account ID to enable tracking
    "google_analytics_account": "UA-116279123-2",
    # colors
    "navigation_button": "#b13a59",
    "navigation_button_hover": "#712b3d",
    "toc_caption": "#b13a59",
    "toc_hover": "#b13a59",
    "table_header_bg": "#ffdce5",
    "table_header_border": "#b13a59",
    "download_button": "#b13a59",
}

edit_on_github_project = "XanaduAI/MrMustard"
edit_on_github_branch = "master/doc"


# the order in which autodoc lists the documented members
autodoc_member_order = "bysource"

# inheritance_diagram graphviz attributes
inheritance_node_attrs = dict(color="lightskyblue1", style="filled")


from custom_directives import CustomGalleryItemDirective, DetailsDirective


def setup(app):
    app.add_directive("customgalleryitem", CustomGalleryItemDirective)
    app.add_directive("details", DetailsDirective)
    app.add_css_file("xanadu_gallery.css")
