"""
Sphinx extension to add ReadTheDocs-style "Edit on GitHub" links to the
sidebar.
Loosely based on https://github.com/astropy/astropy/pull/347
"""

import os
import warnings

__licence__ = "BSD (3 clause)"


def get_github_url(app, view, path):
    return f"https://github.com/{app.config.edit_on_github_project}/{view}/{app.config.edit_on_github_branch}/{path}"


def html_page_context(app, pagename, templatename, context, doctree):
    if templatename != "page.html":
        return

    if not app.config.edit_on_github_project:
        warnings.warn("edit_on_github_project not specified", stacklevel=1)
        return

    if not doctree:
        return

    path = os.path.relpath(doctree.get("source"), app.builder.srcdir)
    show_url = get_github_url(app, "blob", path)
    edit_url = get_github_url(app, "edit", path)

    context["show_on_github_url"] = show_url
    context["edit_on_github_url"] = edit_url


def setup(app):
    app.add_config_value("edit_on_github_project", "", True)
    app.add_config_value("edit_on_github_branch", "master", True)
    app.connect("html-page-context", html_page_context)
