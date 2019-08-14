"""
Utility method which prints system info to help with debugging,
and filing issues on GitHub.
Adapted from :func:`sklearn.show_versions`,
which was adapted from :func:`pandas.show_versions`
"""

# Author: Alexander L. Hayes <hayesall@iu.edu>
# License: MIT

import sys
import importlib


def _get_deps_info():
    """Overview of the installed version of main dependencies
    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    deps = [
        "pip",
        "setuptools",
        "imblearn",
        "sklearn",
        "numpy",
        "scipy",
        "Cython",
        "pandas",
        "keras",
        "tensorflow",
    ]

    def get_version(module):
        return module.__version__

    deps_info = {}

    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            ver = get_version(mod)
            deps_info[modname] = ver
        except ImportError:
            deps_info[modname] = None

    return deps_info


def show_versions(github=False):
    """Print debugging information.
    Parameters
    ----------
    github : bool,
        If true, wrap system info with GitHub markup.
    """

    from sklearn.utils._show_versions import _get_sys_info

    _sys_info = _get_sys_info()
    _deps_info = _get_deps_info()
    _github_markup = (
        "<details>"
        "<summary>System, Dependency Information</summary>\n\n"
        "**System Information**\n\n"
        "{0}\n"
        "**Python Dependencies**\n\n"
        "{1}\n"
        "</details>"
    )

    if github:

        _sys_markup = ""
        _deps_markup = ""

        for k, stat in _sys_info.items():
            _sys_markup += "* {k:<10}: `{stat}`\n".format(k=k, stat=stat)
        for k, stat in _deps_info.items():
            _deps_markup += "* {k:<10}: `{stat}`\n".format(k=k, stat=stat)

        print(_github_markup.format(_sys_markup, _deps_markup))

    else:

        print("\nSystem:")
        for k, stat in _sys_info.items():
            print("{k:>11}: {stat}".format(k=k, stat=stat))

        print("\nPython dependencies:")
        for k, stat in _deps_info.items():
            print("{k:>11}: {stat}".format(k=k, stat=stat))
