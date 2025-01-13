import os
import inspect
import sys
import pytensor
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve() / "scripts"))

# General configuration
# ---------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "ablog",
    "myst_nb",
    "generate_gallery",
    "sphinx_sitemap",
]

# Don't auto-generate summary for class members.
numpydoc_show_class_members = False
autosummary_generate = True
autodoc_typehints = "none"
remove_from_toctrees = ["**/classmethods/*"]


intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

needs_sphinx = "3"

todo_include_todos = True
napoleon_google_docstring = False
napoleon_include_special_with_doc = False

# Add any paths that contain templates here, relative to this directory.
templates_path = [".templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General substitutions.
project = "PyTensor"
copyright = "PyMC Team 2022-present,2020-2021; Aesara Developers, 2021-2022; LISA Lab, 2008--2019"

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#

version = pytensor.__version__
if os.environ.get("READTHEDOCS", False):
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if rtd_version.lower() == "stable":
        version = pytensor.__version__.split("+")[0]
    elif rtd_version.lower() == "latest":
        version = "dev"
    else:
        version = rtd_version
else:
    rtd_version = "local"
# The full version, including alpha/beta/rc tags.
release = version

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = "%B %d, %Y"

# List of documents that shouldn't be included in the build.
# unused_docs = []

# List of directories, relative to source directories, that shouldn't be
# searched for source files.
exclude_dirs = ["images", "scripts", "sandbox"]
exclude_patterns = ['page_footer.md', '**/*.myst.md']

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "py:obj"

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# Options for HTML output
# -----------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pymc_sphinx_theme"
html_logo = "images/PyTensor_RGB.svg"

html_baseurl = "https://pytensor.readthedocs.io"
sitemap_url_scheme = f"{{lang}}{rtd_version}/{{link}}"


html_theme_options = {
    "use_search_override": False,
    "icon_links": [
        {
            "url": "https://github.com/pymc-devs/pytensor/",
            "icon": "fa-brands fa-github",
            "name": "GitHub",
            "type": "fontawesome",
        },
        {
            "url": "https://twitter.com/pymc_devs/",
            "icon": "fa-brands fa-twitter",
            "name": "Twitter",
            "type": "fontawesome",
        },
        {
            "url": "https://www.youtube.com/c/PyMCDevelopers",
            "icon": "fa-brands fa-youtube",
            "name": "YouTube",
            "type": "fontawesome",
        },
        {
            "url": "https://discourse.pymc.io",
            "icon": "fa-brands fa-discourse",
            "name": "Discourse",
            "type": "fontawesome",
        },
    ],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink", "donate"],
    "navbar_start": ["navbar-logo"],
    "article_header_end": ["nb-badges"],
    "article_footer_items": ["rendered_citation.html"],
}
html_context = {
    "github_url": "https://github.com",
    "github_user": "pymc-devs",
    "github_repo": "pytensor",
    "github_version": version if "." in rtd_version else "main",
    "sandbox_repo": f"pymc-devs/pymc-sandbox/{version}",
    "doc_path": "doc",
    "default_mode": "light",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["../_static"]
html_extra_path = ["_thumbnails", 'images', "robots.txt"]
templates_path = [".templates"]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (within the static path) to place at the top of
# the sidebar.
# html_logo = 'images/PyTensor_RGB.svg'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["images", "library/d3viz/examples"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_use_modindex = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, the reST sources are included in the HTML build as _sources/<name>.
# html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = "pytensor_doc"


# Options for the linkcode extension
# ----------------------------------
# Resolve function
# This function is used to populate the (source) links in the API
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        fn = Path(inspect.getsourcefile(obj))
        fn = fn.relative_to(Path(__file__).parent)
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "pytensor/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    import subprocess

    tag = subprocess.Popen(
        ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, universal_newlines=True
    ).communicate()[0][:-1]
    return f"https://github.com/pymc-devs/pytensor/blob/{tag}/{filename}"


# Options for LaTeX output
# ------------------------

latex_elements = {
    # The paper size ('letter' or 'a4').
    # latex_paper_size = 'letter',
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "11pt",
    # Additional stuff for the LaTeX preamble.
    # latex_preamble = '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class
# [howto/manual]).
latex_documents = [
    (
        "index",
        "pytensor.tex",
        "PyTensor Documentation",
        "PyTensor Developers",
        "manual",
    ),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = 'images/PyTensor_RGB.svg'

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_use_modindex = True


# -- MyST config  -------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "substitution",
]
myst_dmath_double_inline = True

citation_code = f"""
```bibtex
@incollection{{citekey,
  author    = "<notebook authors, see above>",
  title     = "<notebook title>",
  editor    = "Pytensor Team",
  booktitle = "Pytensor Examples",
}}
```
"""

myst_substitutions = {
    "pip_dependencies": "{{ extra_dependencies }}",
    "conda_dependencies": "{{ extra_dependencies }}",
    "extra_install_notes": "",
    "citation_code": citation_code,
}

nb_execution_mode = "off"
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

rediraffe_redirects = {
    "index.md": "gallery.md",
}

# -- Bibtex config  -------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"


# -- ablog config  -------------------------------------------------
blog_baseurl = "https://pytensor.readthedocs.io/en/latest/index.html"
blog_title = "Pytensor Examples"
blog_path = "blog"
blog_authors = {
    "contributors": ("Pytensor Contributors", "https://pytensor.readthedocs.io"),
}
blog_default_author = "contributors"
post_show_prev_next = False
fontawesome_included = True
# post_redirect_refresh = 1
# post_auto_image = 1
# post_auto_excerpt = 2

# notfound_urls_prefix = ""
