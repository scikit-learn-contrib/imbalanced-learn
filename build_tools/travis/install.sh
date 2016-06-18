# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Use the miniconda installer for faster download / install of conda
# itself
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo
if [[ ! -f miniconda.sh ]]
   then
   wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
        -O miniconda.sh
   fi
chmod +x miniconda.sh && ./miniconda.sh -b
cd ..
export PATH=/home/travis/miniconda2/bin:$PATH
conda update --yes conda
popd

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
    numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
    libgfortran nomkl
#source activate testenv

# Install nose-timer via pip
pip install nose-timer

# Install libgfortran with conda
conda install --yes libgfortran \
    numpy=1.10.4 scipy=0.17.1 \
    scikit-learn=0.17.1 \
    six=1.10.0 

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

# Build scikit-cycling in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

cd $TRAVIS_BUILD_DIR
python setup.py develop
