# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Abil'
copyright = '2024, Abil developers'
author = 'nanophyto'
release = '25.02.17'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
]

sphinx_gallery_conf = {
    'examples_dirs': 'examples',
    'gallery_dirs': 'auto_examples',
    'filename_pattern': r'.*\.py',
    'image_scrapers': ('matplotlib',),
    'pypandoc': False,
    # Memory optimization options
    'min_reported_time': 0,
    'thumbnail_size': (400, 300),
    'plot_gallery': True,  # Set to False if you don't need plots
    'inspect_global_variables': False,
    'within_subsection_order': lambda folder: [],
    'run_stale_examples': True,  # Only rebuild modified examples
    'matplotlib_animations': False,  # Disable animations to save memory
    'image_srcset': [],  # Disable responsive images
    'remove_config_comments': True,  # Remove config comments to save memory
}

# Additional memory-saving Sphinx settings
nitpicky = False
keep_warnings = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "navbar_end": ["theme-switcher", "version-switcher"],  # Remove "this-page" from navbar
    "secondary_sidebar_items": []  # This removes the "This Page" section
}

