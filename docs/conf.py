from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'graph-neural-network' / 'src'))

project = 'GRACE'
copyright = '2026, Matteo Massari'
author = 'Matteo Massari'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'myst_parser',
    'sphinx_autodoc_typehints',
    "sphinxcontrib.mermaid",
]

exclude_patterns = ['_build', 'it', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_title = 'GRACE'

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
}

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': False,
    'show-inheritance': True,
}

autoclass_content = 'class'
napoleon_google_docstring = True
napoleon_numpy_docstring = False
todo_include_todos = True

autodoc_typehints = 'signature'
typehints_use_signature = True
typehints_document_rtype = True

# Render dollar-delimited math also used by existing docstrings.
myst_enable_extensions = ["dollarmath"]

# Let diagrams use their natural height instead of a fixed, mostly empty canvas.
mermaid_height = "auto"

language = 'en'
templates_path = [str(Path(__file__).resolve().parent / '_templates')]
html_context = {'other_language_prefix': 'it/'}
