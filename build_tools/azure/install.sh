#!/bin/bash

set -e
set -x

UNAMESTR=`uname`

make_conda() {
    TO_INSTALL="$@"
    if [[ "$DISTRIB" == *"mamba"* ]]; then
        mamba create -n $VIRTUALENV --yes $TO_INSTALL
    else
        conda config --show
        conda create -n $VIRTUALENV --yes $TO_INSTALL
    fi
    source activate $VIRTUALENV
}

setup_ccache() {
    echo "Setting up ccache"
    mkdir /tmp/ccache/
    which ccache
    for name in gcc g++ cc c++ x86_64-linux-gnu-gcc x86_64-linux-gnu-c++; do
      ln -s $(which ccache) "/tmp/ccache/${name}"
    done
    export PATH="/tmp/ccache/:${PATH}"
    ccache -M 256M
}

# imports get_dep
source build_tools/shared.sh

if [[ "$DISTRIB" == "conda" || "$DISTRIB" == *"mamba"* ]]; then

    if [[ "$CONDA_CHANNEL" != "" ]]; then
        TO_INSTALL="--override-channels -c $CONDA_CHANNEL"
    else
        TO_INSTALL=""
    fi

    TO_INSTALL="$TO_INSTALL python=$PYTHON_VERSION"
    TO_INSTALL="$TO_INSTALL ccache pip blas[build=$BLAS]"

    TO_INSTALL="$TO_INSTALL $(get_dep numpy $NUMPY_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep scipy $SCIPY_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep scikit-learn $SKLEARN_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep joblib $JOBLIB_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep pandas $PANDAS_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep Pillow $PILLOW_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep matplotlib $MATPLOTLIB_VERSION)"

	make_conda $TO_INSTALL
    setup_ccache

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    sudo add-apt-repository --remove ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install python3-scipy python3-sklearn python3-matplotlib \
        libatlas3-base libatlas-base-dev python3-virtualenv ccache
    python3 -m virtualenv --system-site-packages --python=python3 $VIRTUALENV
    source $VIRTUALENV/bin/activate
    setup_ccache
    python -m pip install $(get_dep joblib $JOBLIB_VERSION)

elif [[ "$DISTRIB" == "debian-32" ]]; then
    apt-get update
    apt-get install -y python3-dev python3-numpy python3-scipy python3-sklearn \
        python3-matplotlib libatlas3-base libatlas-base-dev python3-virtualenv \
        python3-pandas ccache

    python3 -m virtualenv --system-site-packages --python=python3 $VIRTUALENV
    source $VIRTUALENV/bin/activate
    setup_ccache
    python -m pip install $(get_dep joblib $JOBLIB_VERSION)

elif [[ "$DISTRIB" == "conda-pip-latest" ]]; then
    # Since conda main channel usually lacks behind on the latest releases,
    # we use pypi to test against the latest releases of the dependencies.
    # conda is still used as a convenient way to install Python and pip.
    make_conda "ccache python=$PYTHON_VERSION"
    setup_ccache
    python -m pip install -U pip

    python -m pip install pandas matplotlib

elif [[ "$DISTRIB" == "conda-pip-scipy-dev" ]]; then
    make_conda "ccache python=$PYTHON_VERSION"
    python -m pip install -U pip
    echo "Installing numpy and scipy master wheels"
    dev_anaconda_url=https://pypi.anaconda.org/scipy-wheels-nightly/simple
    pip install --pre --upgrade --timeout=60 --extra-index $dev_anaconda_url numpy pandas scipy scikit-learn
    setup_ccache
    echo "Installing joblib master"
    pip install https://github.com/joblib/joblib/archive/master.zip
    echo "Installing pillow master"
    pip install https://github.com/python-pillow/Pillow/archive/main.zip
fi

python -m pip install $(get_dep threadpoolctl $THREADPOOLCTL_VERSION) \
                      $(get_dep pytest $PYTEST_VERSION) \
                      $(get_dep pytest-xdist $PYTEST_XDIST_VERSION)

if [[ "$COVERAGE" == "true" ]]; then
    python -m pip install codecov pytest-cov
fi

if [[ "$PYTEST_XDIST_VERSION" != "none" ]]; then
    python -m pip install pytest-xdist
fi

if [[ "$TEST_DOCSTRINGS" == "true" ]]; then
    # numpydoc requires sphinx
    python -m pip install sphinx
    python -m pip install numpydoc
fi

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "\
try:
    import pandas
    print('pandas %s' % pandas.__version__)
except ImportError:
    print('pandas not installed')
"

python -m pip list
if [[ "$DISTRIB" == "conda-pip-latest" ]]; then
    # Check that pip can automatically build scikit-learn with the build
    # dependencies specified in pyproject.toml using an isolated build
    # environment:
    pip install --verbose --editable .
fi
ccache -s
