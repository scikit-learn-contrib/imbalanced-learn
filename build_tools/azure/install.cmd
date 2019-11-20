@rem https://github.com/numba/numba/blob/master/buildscripts/incremental/setup_conda_environment.cmd
@rem The cmd /C hack circumvents a regression where conda installs a conda.bat
@rem script in non-root environments.
set CONDA_INSTALL=cmd /C conda install -q -y
set PIP_INSTALL=pip install -q

@echo on

IF "%PYTHON_ARCH%"=="64" (
    @rem Deactivate any environment
    call deactivate
    @rem Clean up any left-over from a previous build
    conda remove --all -q -y -n %VIRTUALENV%
    conda create -n %VIRTUALENV% -q -y python=%PYTHON_VERSION% numpy scipy cython wheel joblib git

    call activate %VIRTUALENV%

    IF "%PYTEST_VERSION%"=="*" (
        pip install pytest
    ) else (
        pip install pytest==%PYTEST_VERSION%
    )
    pip install pytest-xdist
) else (
    pip install numpy scipy cython pytest wheel pillow joblib
)
if "%COVERAGE%" == "true" (
    pip install coverage codecov pytest-cov
)
python --version
pip --version

pip install git+https://github.com/scikit-learn/scikit-learn.git

@rem Install the build and runtime dependencies of the project.
python setup.py bdist_wheel bdist_wininst

@rem Install the generated wheel package to test it
pip install --pre --no-index --find-links dist\ imbalanced-learn

if %errorlevel% neq 0 exit /b %errorlevel%
