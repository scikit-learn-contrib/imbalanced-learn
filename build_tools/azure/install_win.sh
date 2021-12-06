#!/bin/bash

set -e
set -x

if [[ "$PYTHON_ARCH" == "64" ]]; then
    conda create -n $VIRTUALENV -q -y python=$PYTHON_VERSION numpy scipy scikit-learn matplotlib wheel pillow joblib

    source activate $VIRTUALENV

    pip install threadpoolctl

    if [[ "$PYTEST_VERSION" == "*" ]]; then
        pip install pytest
    else
        pip install pytest==$PYTEST_VERSION
    fi
else
    pip install numpy scipy scikit-learn pytest wheel pillow joblib threadpoolctl
fi

if [[ "$PYTEST_XDIST_VERSION" != "none" ]]; then
    pip install pytest-xdist
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage codecov pytest-cov
fi

python --version
pip --version

python -m pip list
pip install --verbose --editable .
