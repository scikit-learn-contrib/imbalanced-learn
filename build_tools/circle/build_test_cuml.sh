#!/usr/bin/env bash
set -x
set -e

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
    deactivate
fi

# Install dependencies with miniconda
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
export PATH="$MINICONDA_PATH/bin:$PATH"
conda update --yes --quiet conda

# imports get_dep
source build_tools/shared.sh

# packaging won't be needed once setuptools starts shipping packaging>=17.0
mamba create -n $CONDA_ENV_NAME --yes --quiet \
    -c rapidsai -c rapidsai-nightly -c nvidia -c conda-forge \
    python="${PYTHON_VERSION:-*}" \
    "$(get_dep numpy $NUMPY_VERSION)" \
    "$(get_dep scipy $SCIPY_VERSION)" \
    "$(get_dep scikit-learn $SKLEARN_VERSION)" \
    "$(get_dep cuml $CUML_VERSION)" \
    "$(get_dep cudatoolkit $CUDATOOLKIT_VERSION)"
source activate $CONDA_ENV_NAME
python -m pip install pytest coverage pytest-cov pytest-xdist

# Build and install imbalanced-learn in dev mode
ls -l
pip install -e . --no-build-isolation

# Test the install
pytest -v --cov=imblearn imblearn -n 2
