#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# Travis clone pydicom/pydicom repository in to a local repository.

set -e

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

export CC=/usr/lib/ccache/gcc
export CXX=/usr/lib/ccache/g++
# Useful for debugging how ccache is used
# export CCACHE_LOGFILE=/tmp/ccache.log
# ~60M is used by .ccache when compiling from scratch at the time of writing
ccache --max-size 100M --show-stats

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Install miniconda
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O miniconda.sh
    MINICONDA_PATH=/home/travis/miniconda
    chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
    export PATH=$MINICONDA_PATH/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip
    source activate testenv
    conda install --yes numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION

    if [[ $PYTHON_VERSION == "3.6" ]]; then
        # Tensorflow is not available in Python 3.7 yet.
        conda install --yes pandas keras tensorflow
        KERAS_BACKEND=tensorflow
        python -c "import keras.backend"
        sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
    fi

    if [[ "$SKLEARN_VERSION" == "master" ]]; then
        conda install --yes cython
        pip install -U git+https://github.com/scikit-learn/scikit-learn.git
    else
        conda install --yes scikit-learn=$SKLEARN_VERSION
    fi

    conda install --yes pytest pytest-cov
    pip install codecov

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    # At the time of writing numpy 1.9.1 is included in the travis
    # virtualenv but we want to use the numpy installed through apt-get
    # install.
    deactivate
    # Create a new virtualenv using system site packages for python, numpy
    virtualenv --system-site-packages testvenv
    source testvenv/bin/activate

    pip install scikit-learn
    pip install pandas keras tensorflow
    pip install pytest pytest-cov codecov sphinx numpydoc

fi

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

pip install -e .
ccache --show-stats
# Useful for debugging how ccache is used
# cat $CCACHE_LOGFILE
