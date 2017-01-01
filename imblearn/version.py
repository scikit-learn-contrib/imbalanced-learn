"""
``imbalanced-learn`` is a set of python methods to deal with imbalanced
datset in machine learning and pattern recognition.
"""
# Based on NiLearn package
# License: simplified BSD

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
# X.Y
# X.Y.Z # For bugfix releases
#
# Admissible pre-release markers:
# X.YaN # Alpha release
# X.YbN # Beta release
# X.YrcN # Release Candidate
# X.Y # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '0.2.1'

_IMBALANCED_DATASET_INSTALL_MSG = 'See %s for installation information.' % (
    'http://contrib.scikit-learn.org/imbalanced-learn/install.html')

# This is a tuple to preserve order, so that dependencies are checked
#   in some meaningful order (more => less 'core').  We avoid using
#   collections.OrderedDict to preserve Python 2.6 compatibility.
REQUIRED_MODULE_METADATA = (('numpy', {
    'min_version': '1.10.4',
    'required_at_installation': True,
    'install_info': _IMBALANCED_DATASET_INSTALL_MSG
}), ('scipy', {
    'min_version': '0.17.0',
    'required_at_installation': True,
    'install_info': _IMBALANCED_DATASET_INSTALL_MSG
}), ('sklearn', {
    'min_version': '0.17.1',
    'required_at_installation': True,
    'install_info': _IMBALANCED_DATASET_INSTALL_MSG
}))


def _import_module_with_version_check(module_name,
                                      minimum_version,
                                      install_info=None):
    """Check that module is installed with a recent enough version
    """
    from distutils.version import LooseVersion

    try:
        module = __import__(module_name)
    except ImportError as exc:
        user_friendly_info = ('Module "{0}" could not be found. {1}').format(
            module_name, install_info or 'Please install it properly to use'
            ' imbalanced-learn.')
        exc.args += (user_friendly_info, )
        raise

    # Avoid choking on modules with no __version__ attribute
    module_version = getattr(module, '__version__', '0.0.0')

    version_too_old = (
        not LooseVersion(module_version) >= LooseVersion(minimum_version))

    if version_too_old:
        message = ('A {module_name} version of at least {minimum_version} '
                   'is required to use imbalanced-learn. {module_version} was '
                   'found. Please upgrade {module_name}').format(
                       module_name=module_name,
                       minimum_version=minimum_version,
                       module_version=module_version)

        raise ImportError(message)

    return module


def _check_module_dependencies(is_imbalanced_dataset_installing=False):
    """Throw an exception if imbalanced-learn dependencies are not installed.
    Parameters
    ----------
    is_imbalanced_dataset_installing: boolean
        if True, only error on missing packages that cannot be auto-installed.
        if False, error on any missing package.
    Throws
    -------
    ImportError
    """

    for (module_name, module_metadata) in REQUIRED_MODULE_METADATA:
        if not (is_imbalanced_dataset_installing and
                not module_metadata['required_at_installation']):
            # Skip check only when installing and it's a module that
            # will be auto-installed.
            _import_module_with_version_check(
                module_name=module_name,
                minimum_version=module_metadata['min_version'],
                install_info=module_metadata.get('install_info'))
