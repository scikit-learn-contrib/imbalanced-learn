#!/bin/bash

set -e
set -x

if [[ "$PYTHON_ARCH" == "64" ]]; then
    conda create -n $VIRTUALENV -q -y python=$PYTHON_VERSION numpy scipy scikit-learn cython matplotlib wheel pillow joblib

    source activate $VIRTUALENV

    pip install threadpoolctl

    if [[ "$PYTEST_VERSION" == "*" ]]; then
        pip install pytest
    else
        pip install pytest==$PYTEST_VERSION
    fi
else
    pip install numpy scipy scikit-learn cython pytest wheel pillow joblib threadpoolctl
fi

if [[ "$PYTEST_XDIST_VERSION" != "none" ]]; then
    pip install pytest-xdist
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage codecov pytest-cov
fi

python --version
pip --version

# Build imbalanced-learn
python setup.py bdist_wheel

# Install the generated wheel package to test it
pip install --pre --no-index --find-links dist imbalanced-learn
