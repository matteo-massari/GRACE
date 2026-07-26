import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

project = 'Graph Recommendation System'
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

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_title = 'Graph Recommendation System'

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
