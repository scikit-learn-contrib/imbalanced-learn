#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# Travis clone pydicom/pydicom repository in to a local repository.

set -e

export CC=/usr/lib/ccache/gcc
export CXX=/usr/lib/ccache/g++
# Useful for debugging how ccache is used
# export CCACHE_LOGFILE=/tmp/ccache.log
# ~60M is used by .ccache when compiling from scratch at the time of writing
ccache --max-size 100M --show-stats

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O miniconda.sh
MINICONDA_PATH=/home/travis/miniconda
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
export PATH=$MINICONDA_PATH/bin:$PATH

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n testenv --yes python=$PYTHON_VERSION pip
source activate testenv

pip install --upgrade pip setuptools
echo "Installing numpy, scipy, and pandas master wheels"
dev_url=https://pypi.anaconda.org/scipy-wheels-nightly/simple
pip install --pre --upgrade --timeout=60 -f $dev_url numpy scipy pandas
echo "Installing joblib master"
pip install https://github.com/joblib/joblib/archive/master.zip
echo "Installing scikit-learn master"
pip install --pre --extra-index $dev_url scikit-learn

conda install --yes pytest pytest-cov
pip install codecov
pip install -U git+https://github.com/numpy/numpydoc.git

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

pip install -e .
ccache --show-stats
# Useful for debugging how ccache is used
# cat $CCACHE_LOGFILE
