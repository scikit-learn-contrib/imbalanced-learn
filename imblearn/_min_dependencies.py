"""All minimum dependencies for imbalanced-learn."""
import argparse

# numpy scipy and cython should by in sync with pyproject.toml
# We pinned PyWavelet (a scikit-image dependence) to 1.1.1 in the minimum
# documentation CI builds that is the latest version that support our
# minimum NumPy version required. If PyWavelets 1.2+ is installed, it would
# require NumPy 1.17+ that trigger a bug with Pandas 0.25:
# https://github.com/numpy/numpy/issues/18355#issuecomment-774610226
# When upgrading NumPy, we can unpin PyWavelets but we need to update the
# minimum version of Pandas >= 1.0.5.
NUMPY_MIN_VERSION = "1.14.6"
SCIPY_MIN_VERSION = "1.1.0"
PANDAS_MIN_VERSION = "0.25.0"
SKLEARN_MIN_VERSION = "1.0.1"
TENSORFLOW_MIN_VERSION = "2.4.3"
KERAS_MIN_VERSION = "2.4.3"
JOBLIB_MIN_VERSION = "0.11"
THREADPOOLCTL_MIN_VERSION = "2.0.0"
PYTEST_MIN_VERSION = "5.0.1"


# 'build' and 'install' is included to have structured metadata for CI.
# It will NOT be included in setup's extras_require
# The values are (version_spec, comma separated tags)
dependent_packages = {
    "numpy": (NUMPY_MIN_VERSION, "install"),
    "scipy": (SCIPY_MIN_VERSION, "install"),
    "scikit-learn": (SKLEARN_MIN_VERSION, "install"),
    "joblib": (JOBLIB_MIN_VERSION, "install"),
    "threadpoolctl": (THREADPOOLCTL_MIN_VERSION, "install"),
    "pandas": (PANDAS_MIN_VERSION, "optional, docs, examples, tests"),
    "tensorflow": (TENSORFLOW_MIN_VERSION, "optional, docs, examples, tests"),
    "keras": (KERAS_MIN_VERSION, "optional, docs, examples, tests"),
    "matplotlib": ("2.2.3", "docs, examples"),
    "seaborn": ("0.9.0", "docs, examples"),
    "memory_profiler": ("0.57.0", "docs"),
    "pytest": (PYTEST_MIN_VERSION, "tests"),
    "pytest-cov": ("2.9.0", "tests"),
    "flake8": ("3.8.2", "tests"),
    "black": ("21.6b0", "tests"),
    "mypy": ("0.770", "tests"),
    "sphinx": ("4.2.0", "docs"),
    "sphinx-gallery": ("0.7.0", "docs"),
    "numpydoc": ("1.0.0", "docs"),
    "sphinxcontrib-bibtex": ("2.4.1", "docs"),
    "pydata-sphinx-theme": ("0.7.2", "docs"),
}


# create inverse mapping for setuptools
tag_to_packages: dict = {
    extra: [] for extra in ["install", "optional", "docs", "examples", "tests"]
}
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))


# Used by CI to get the min dependencies
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies for a package")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)
