#!/bin/bash

set -e
set -x

UNAMESTR=`uname`

make_conda() {
    conda update -yq conda
    TO_INSTALL="$@"
    if [[ "$DISTRIB" == *"mamba"* ]]; then
        mamba create -n $VIRTUALENV --yes $TO_INSTALL
    else
        conda config --show
        conda create -n $VIRTUALENV --yes $TO_INSTALL
    fi
    source activate $VIRTUALENV
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
    TO_INSTALL="$TO_INSTALL pip blas[build=$BLAS]"

    TO_INSTALL="$TO_INSTALL $(get_dep numpy $NUMPY_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep scipy $SCIPY_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep scikit-learn $SKLEARN_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep joblib $JOBLIB_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep pandas $PANDAS_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep Pillow $PILLOW_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep matplotlib $MATPLOTLIB_VERSION)"

	make_conda $TO_INSTALL

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    sudo add-apt-repository --remove ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install python3-scipy python3-sklearn python3-matplotlib \
        libatlas3-base libatlas-base-dev python3-virtualenv
    python3 -m virtualenv --system-site-packages --python=python3 $VIRTUALENV
    source $VIRTUALENV/bin/activate
    python -m pip install $(get_dep joblib $JOBLIB_VERSION)

elif [[ "$DISTRIB" == "debian-32" ]]; then
    apt-get update
    apt-get install -y python3-dev python3-numpy python3-scipy python3-sklearn \
        python3-matplotlib libatlas3-base libatlas-base-dev python3-virtualenv \
        python3-pandas

    python3 -m virtualenv --system-site-packages --python=python3 $VIRTUALENV
    source $VIRTUALENV/bin/activate
    python -m pip install $(get_dep joblib $JOBLIB_VERSION)

elif [[ "$DISTRIB" == "conda-pip-latest" ]]; then
    # Since conda main channel usually lacks behind on the latest releases,
    # we use pypi to test against the latest releases of the dependencies.
    # conda is still used as a convenient way to install Python and pip.
    make_conda "python=$PYTHON_VERSION"
    python -m pip install -U pip

    python -m pip install pandas matplotlib
    python -m pip install scikit-learn

elif [[ "$DISTRIB" == "conda-pip-latest-tensorflow" ]]; then
    make_conda "python=$PYTHON_VERSION"
    python -m pip install -U pip

    python -m pip install numpy scipy scikit-learn pandas tensorflow

elif [[ "$DISTRIB" == "conda-latest-tensorflow" ]]; then
    make_conda "python=$PYTHON_VERSION numpy scipy scikit-learn pandas tensorflow"

elif [[ "$DISTRIB" == "conda-minimum-tensorflow" ]]; then
    TO_INSTALL="python=$PYTHON_VERSION"
    TO_INSTALL="$TO_INSTALL $(get_dep numpy $NUMPY_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep scipy $SCIPY_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep scikit-learn $SKLEARN_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep pandas $PANDAS_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep tensorflow $TENSORFLOW_VERSION)"
    make_conda $TO_INSTALL

elif [[ "$DISTRIB" == "conda-pip-latest-keras" ]]; then
    make_conda "python=$PYTHON_VERSION"
    python -m pip install -U pip

    python -m pip install numpy scipy scikit-learn pandas keras

elif [[ "$DISTRIB" == "conda-latest-keras" ]]; then
    make_conda "python=$PYTHON_VERSION numpy scipy scikit-learn pandas keras"

elif [[ "$DISTRIB" == "conda-minimum-keras" ]]; then
    TO_INSTALL="python=$PYTHON_VERSION"
    TO_INSTALL="$TO_INSTALL $(get_dep numpy $NUMPY_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep scipy $SCIPY_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep scikit-learn $SKLEARN_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep pandas $PANDAS_VERSION)"
    TO_INSTALL="$TO_INSTALL $(get_dep keras $KERAS_VERSION)"
    make_conda $TO_INSTALL

elif [[ "$DISTRIB" == "conda-pip-scipy-dev" ]]; then
    make_conda "python=$PYTHON_VERSION"
    python -m pip install -U pip
    echo "Installing numpy and scipy master wheels"
    dev_anaconda_url=https://pypi.anaconda.org/scipy-wheels-nightly/simple
    pip install --pre --upgrade --timeout=60 --extra-index $dev_anaconda_url numpy pandas scipy scikit-learn
    echo "Installing joblib master"
    pip install https://github.com/joblib/joblib/archive/master.zip
    echo "Installing tensorflow master"
    pip install tf-nightly
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
pip install --verbose --editable .
