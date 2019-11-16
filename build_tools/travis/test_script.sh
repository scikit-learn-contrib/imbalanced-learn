#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

run_tests(){
    # Get into a temp directory to run test from the installed scikit learn and
    # check if we do not leave artifacts
    mkdir -p $TEST_DIR
    # We need the setup.cfg for the pytest settings
    cp setup.cfg $TEST_DIR
    cd $TEST_DIR

    python --version
    python -c "import numpy; print('numpy %s' % numpy.__version__)"
    python -c "import scipy; print('scipy %s' % scipy.__version__)"
    python -c "import multiprocessing as mp; print('%d CPUs' % mp.cpu_count())"

    pytest --cov=$MODULE -r sx --pyargs $MODULE

    # Test doc
    cd $OLDPWD
    if [[ "$TEST_DOC" == "true" ]]; then
        make test-doc
    fi

    # Validate numpydoc style
    if [[ "$TEST_NUMPYDOC" == "true" ]]; then
        pytest -vsl maint_tools/test_docstring.py
    fi
}

if [[ "$SKIP_TESTS" != "true" ]]; then
    run_tests
fi

# Is directory still empty ?
ls -ltra $TEST_DIR
ls -ltra $TRAVIS_BUILD_DIR
cp $TEST_DIR/.coverage $TRAVIS_BUILD_DIR
