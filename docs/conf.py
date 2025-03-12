# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Abil'
copyright = '2024, Abil developers'
author = 'nanophyto'
release = '25.03.12'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc', 
	'sphinx.ext.coverage', 
	'sphinx.ext.napoleon',
    "sphinx_design"

]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "navbar_end": ["theme-switcher", "version-switcher"],  # Remove "this-page" from navbar
    "secondary_sidebar_items": []  # This removes the "This Page" section
}

