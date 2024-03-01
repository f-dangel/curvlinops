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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "CurvLinOps"
copyright = "2022, F. Dangel, L. Tatzel, R. Eschenhagen"
author = "F. Dangel, L. Tatzel, R. Eschenhagen"

# The full version, including alpha/beta/rc tags
release = "1.2.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    'sphinx.ext.viewcode', # show source code links
]

# -- Intersphinx config -----------------------------------------------------

intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None),
    "backpack": ("https://docs.backpack.pt/en/master", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Sphinx Gallery config ---------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": [
        "../examples/basic_usage",
        # "../examples/use_cases",
    ],  # path to your example scripts
    "gallery_dirs": [
        "basic_usage",
        # "use_cases",
    ],  # path to where to save gallery generated output
    "default_thumb_file": "assets/logo.png",
    "filename_pattern": "example",
    "matplotlib_animations": True,
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "assets/logo.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
