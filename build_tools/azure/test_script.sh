#!/bin/bash

set -e

if [[ "$DISTRIB" =~ ^conda.* ]]; then
    source activate $VIRTUALENV
elif [[ "$DISTRIB" == "ubuntu" ]] || [[ "$DISTRIB" == "debian-32" ]]; then
    source $VIRTUALENV/bin/activate
fi

mkdir -p $TEST_DIR
cp setup.cfg $TEST_DIR
cd $TEST_DIR

# python -c "import joblib; print(f'Number of cores (physical): \
# {joblib.cpu_count()} ({joblib.cpu_count(only_physical_cores=True)})')"
# python -c "import sklearn; sklearn.show_versions()"
python -c "import imblearn; imblearn.show_versions()"

if ! command -v conda &> /dev/null
then
    pip list
else
    # conda list provides more info than pip list (when available)
    conda list
fi

TEST_CMD="python -m pytest --showlocals --durations=20 --junitxml=$JUNITXML"

if [[ "$COVERAGE" == "true" ]]; then
    # Note: --cov-report= is used to disable to long text output report in the
    # CI logs. The coverage data is consolidated by codecov to get an online
    # web report across all the platforms so there is no need for this text
    # report that otherwise hides the test failures and forces long scrolls in
    # the CI logs.
    export COVERAGE_PROCESS_START="$BUILD_SOURCESDIRECTORY/.coveragerc"
    TEST_CMD="$TEST_CMD --cov-config='$COVERAGE_PROCESS_START' --cov imblearn --cov-report="
fi

if [[ "$CHECK_WARNINGS" == "true" ]]; then
    # numpy's 1.19.0's tostring() deprecation is ignored until scipy and joblib removes its usage
    TEST_CMD="$TEST_CMD -Werror::DeprecationWarning -Werror::FutureWarning -Wignore:tostring:DeprecationWarning"

    # numpy's 1.20's np.object deprecationg is ignored until tensorflow removes its usage
    TEST_CMD="$TEST_CMD -Wignore:\`np.object\`:DeprecationWarning"

    # Python 3.10 deprecates disutils and is imported by numpy interally during import time
    TEST_CMD="$TEST_CMD -Wignore:The\ distutils:DeprecationWarning"

    # Workaround for https://github.com/pypa/setuptools/issues/2885
    TEST_CMD="$TEST_CMD -Wignore:Creating\ a\ LegacyVersion:DeprecationWarning"
fi

if [[ "$PYTEST_XDIST_VERSION" != "none" ]]; then
    TEST_CMD="$TEST_CMD -n$CPU_COUNT"
fi

if [[ "$SHOW_SHORT_SUMMARY" == "true" ]]; then
    TEST_CMD="$TEST_CMD -ra"
fi

set -x
eval "$TEST_CMD --pyargs imblearn"
set +x
