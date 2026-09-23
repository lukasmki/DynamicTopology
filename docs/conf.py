# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DynamicTopology"
copyright = "2026, Lukas Kim"
author = "Lukas Kim"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The docstrings are written as prose with `backticked` identifiers rather than
# as reST cross-references, so render single backticks as inline code instead
# of the reST default (italic "interpreted text").
default_role = "code"

# -- autodoc -----------------------------------------------------------------
# The package is imported from the uv environment (`uv sync`), so no sys.path
# manipulation is needed; build with `uv run make -C docs html`.

autodoc_default_options = {
    "members": True,
    # Several core classes (`System`, `Topology`, `Reaction`, ...) carry their
    # documentation in comments rather than a class docstring.
    "undoc-members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "signature"
# Undocumented overrides would otherwise show the parent's docstring -- ASE's
# `Calculator.__init__`, which lists parameters these classes do not take.
autodoc_inherit_docstrings = False
autodoc_class_signature = "separated"
autosummary_generate = False
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "ase": ("https://docs.ase-lib.org", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "DynamicTopology"
html_static_path = ["_static"]


# -- Docstring preprocessing -------------------------------------------------

_LIST_ITEM = re.compile(r"\s*([-*+]|\d+[.)])\s")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def literal_blocks(app, what, name, obj, options, lines):
    """Render the docstrings' indented equation blocks verbatim.

    The source docstrings follow a plain-text convention: a paragraph followed
    by an indented block of equations or a table, e.g.

        K_ij = sum_n' [erf(gamma |r_ij + n|) ...] / |r_ij + n|

    reST would parse that as a block quote, reading `|...|` as a substitution.
    Ending the introducing line with `::` (or adding a bare `::` paragraph when
    it does not end in a colon) makes it a literal block instead.  Indented
    bullet lists are left alone.
    """
    for i in reversed(range(len(lines) - 2)):
        line = lines[i].rstrip()
        following = lines[i + 2]
        if not line or line.endswith("::") or lines[i + 1].strip():
            continue
        if not following.strip() or _indent(following) <= _indent(line):
            continue
        if _LIST_ITEM.match(following) or following.lstrip().startswith(".."):
            continue
        if line.endswith(":"):
            lines[i] = line + ":"
        else:
            lines[i + 1 : i + 2] = ["", " " * _indent(line) + "::", ""]


def setup(app):
    app.connect("autodoc-process-docstring", literal_blocks)
