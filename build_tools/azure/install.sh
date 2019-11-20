#!/bin/bash

set -e

UNAMESTR=`uname`

make_conda() {
    TO_INSTALL="$@"
    conda create -n $VIRTUALENV --yes $TO_INSTALL
    source activate $VIRTUALENV
}

version_ge() {
    # The two version numbers are seperated with a new line is piped to sort
    # -rV. The -V activates for version number sorting and -r sorts in
    # decending order. If the first argument is the top element of the sort, it
    # is greater than or equal to the second argument.
    test "$(printf "${1}\n${2}" | sort -rV | head -n 1)" == "$1"
}

if [[ "$DISTRIB" == "conda" ]]; then

    TO_INSTALL="python=$PYTHON_VERSION pip \
                numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
                joblib=$JOBLIB_VERSION"

    if [[ "$INSTALL_MKL" == "true" ]]; then
        TO_INSTALL="$TO_INSTALL mkl"
    else
        TO_INSTALL="$TO_INSTALL nomkl"
    fi

    if [[ -n "$PANDAS_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL pandas=$PANDAS_VERSION"
    fi

    if [[ -n "$KERAS_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL keras=$KERAS_VERSION tensorflow=1"
        KERAS_BACKEND=tensorflow
        python -c "import keras.backend"
        sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
    fi

    if [[ -n "$TENSORFLOW_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL tensorflow=$TENSORFLOW_VERSION"
    fi

    make_conda $TO_INSTALL

    if [[ "$PYTEST_VERSION" == "*" ]]; then
        python -m pip install pytest
    else
        python -m pip install pytest=="$PYTEST_VERSION"
    fi

    if [[ "$PYTHON_VERSION" == "*" ]]; then
        python -m pip install pytest-xdist
    fi

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    sudo add-apt-repository --remove ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install python3-scipy libatlas3-base libatlas-base-dev libatlas-dev python3-virtualenv
    python3 -m virtualenv --system-site-packages --python=python3 $VIRTUALENV
    source $VIRTUALENV/bin/activate
    python -m pip install --pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn
    python -m pip install pandas
    python -m pip install pytest==$PYTEST_VERSION pytest-cov joblib==$JOBLIB_VERSION
elif [[ "$DISTRIB" == "ubuntu-32" ]]; then
    apt-get update
    apt-get install -y python3-dev python3-scipy libatlas3-base libatlas-base-dev libatlas-dev python3-virtualenv
    python3 -m virtualenv --system-site-packages --python=python3 $VIRTUALENV
    source $VIRTUALENV/bin/activate
    python -m pip install --pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn
    python -m pip install pandas
    python -m pip install pytest==$PYTEST_VERSION pytest-cov joblib==$JOBLIB_VERSION
elif [[ "$DISTRIB" == "conda-pip-latest" ]]; then
    # Since conda main channel usually lacks behind on the latest releases,
    # we use pypi to test against the latest releases of the dependencies.
    # conda is still used as a convenient way to install Python and pip.
    make_conda "python=$PYTHON_VERSION"
    python -m pip install -U pip
    python -m pip install numpy scipy joblib
    python -m pip install pytest==$PYTEST_VERSION pytest-cov pytest-xdist
    python -m pip install pandas
fi

if [[ "$COVERAGE" == "true" ]]; then
    python -m pip install coverage codecov pytest-cov
fi

if [[ "$TEST_DOCSTRINGS" == "true" ]]; then
    python -m pip install sphinx
    pythong -m pip install -U git+https://github.com/numpy/numpydoc.git
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

# Use setup.py instead of `pip install -e .` to be able to pass the -j flag
# to speed-up the building multicore CI machines.
python setup.py develop
