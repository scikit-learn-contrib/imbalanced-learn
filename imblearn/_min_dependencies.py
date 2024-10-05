"""All minimum dependencies for imbalanced-learn."""
import argparse

NUMPY_MIN_VERSION = "1.24.3"
SCIPY_MIN_VERSION = "1.10.1"
PANDAS_MIN_VERSION = "1.5.3"
SKLEARN_MIN_VERSION = "1.2.2"
TENSORFLOW_MIN_VERSION = "2.13.1"
KERAS_MIN_VERSION = "3.0.5"
JOBLIB_MIN_VERSION = "1.1.1"
THREADPOOLCTL_MIN_VERSION = "2.0.0"
PYTEST_MIN_VERSION = "7.2.2"

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
    "matplotlib": ("3.7.3", "docs, examples"),
    "seaborn": ("0.12.2", "docs, examples"),
    "memory_profiler": ("0.61.0", "docs"),
    "numpydoc": ("1.5.0", "docs, tests"),
    "pytest": (PYTEST_MIN_VERSION, "tests"),
    "pytest-cov": ("4.1.0", "tests"),
    "pytest-xdist": ("3.5.0", "tests"),
    "black": ("23.3.0", "tests"),
    "ruff": ("0.4.8", "tests"),
    "mypy": ("1.3.0", "tests"),
    "sphinx": ("8.0.2", "docs"),
    "sphinx-gallery": ("0.13.0", "docs"),
    "sphinx-copybutton": ("0.5.2", "docs"),
    "sphinxcontrib-bibtex": ("2.4.1", "docs"),
    "pydata-sphinx-theme": ("0.15.4", "docs"),
    "sphinx-design": ("0.6.1", "docs"),
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
