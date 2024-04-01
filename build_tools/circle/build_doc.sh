#!/usr/bin/env bash
set -x
set -e

# Decide what kind of documentation build to run, and run it.
#
# If the last commit message has a "[doc skip]" marker, do not build
# the doc. On the contrary if a "[doc build]" marker is found, build the doc
# instead of relying on the subsequent rules.
#
# We always build the documentation for jobs that are not related to a specific
# PR (e.g. a merge to master or a maintenance branch).
#
# If this is a PR, do a full build if there are some files in this PR that are
# under the "doc/" or "examples/" folders, otherwise perform a quick build.
#
# If the inspection of the current commit fails for any reason, the default
# behavior is to quick build the documentation.

get_build_type() {
    if [ -z "$CIRCLE_SHA1" ]
    then
	echo SKIP: undefined CIRCLE_SHA1
	return
    fi
    commit_msg=$(git log --format=%B -n 1 $CIRCLE_SHA1)
    if [ -z "$commit_msg" ]
    then
	echo QUICK BUILD: failed to inspect commit $CIRCLE_SHA1
	return
    fi
    if [[ "$commit_msg" =~ \[doc\ skip\] ]]
    then
	echo SKIP: [doc skip] marker found
	return
    fi
    if [[ "$commit_msg" =~ \[doc\ quick\] ]]
    then
	echo QUICK: [doc quick] marker found
	return
    fi
    if [[ "$commit_msg" =~ \[doc\ build\] ]]
    then
	echo BUILD: [doc build] marker found
	return
    fi
    if [ -z "$CI_PULL_REQUEST" ]
    then
	echo BUILD: not a pull request
	return
    fi
    git_range="origin/master...$CIRCLE_SHA1"
    git fetch origin master >&2 || (echo QUICK BUILD: failed to get changed filenames for $git_range; return)
    filenames=$(git diff --name-only $git_range)
    if [ -z "$filenames" ]
    then
	echo QUICK BUILD: no changed filenames for $git_range
	return
    fi
    if echo "$filenames" | grep -q -e ^examples/
    then
	echo BUILD: detected examples/ filename modified in $git_range: $(echo "$filenames" | grep -e ^examples/ | head -n1)
	return
    fi
    echo QUICK BUILD: no examples/ filename modified in $git_range:
    echo "$filenames"
}

build_type=$(get_build_type)
if [[ "$build_type" =~ ^SKIP ]]
then
    exit 0
fi

make_args=html
make_args="SPHINXOPTS=-T $make_args"  # show full traceback on exception

# Installing required system packages to support the rendering of math
# notation in the HTML documentation and to optimize the image files
sudo -E apt-get -yq update --allow-releaseinfo-change
sudo -E apt-get -yq remove texlive-binaries --purge
sudo -E apt-get -yq --no-install-suggests --no-install-recommends \
install dvipng texlive-latex-base texlive-latex-extra \
    texlive-latex-recommended texlive-fonts-recommended \
    latexmk gsfonts zip optipng

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
    deactivate
fi

MAMBAFORGE_PATH=$HOME/mambaforge
# Install dependencies with mamba
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
    -O mambaforge.sh
chmod +x mambaforge.sh && ./mambaforge.sh -b -p $MAMBAFORGE_PATH
export PATH="$MAMBAFORGE_PATH/bin:$PATH"
mamba update --yes --quiet conda

# imports get_dep
source build_tools/shared.sh

# packaging won't be needed once setuptools starts shipping packaging>=17.0
mamba create -n $CONDA_ENV_NAME --yes --quiet \
    python="${PYTHON_VERSION:-*}" \
    "$(get_dep numpy $NUMPY_VERSION)" \
    "$(get_dep scipy $SCIPY_VERSION)" \
    "$(get_dep scikit-learn $SKLEARN_VERSION)" \
    "$(get_dep matplotlib $MATPLOTLIB_VERSION)" \
    "$(get_dep sphinx $SPHINX_VERSION)" \
    "$(get_dep pandas $PANDAS_VERSION)" \
    "$(get_dep sphinx-gallery $SPHINX_GALLERY_VERSION)" \
    "$(get_dep numpydoc $NUMPYDOC_VERSION)" \
    "$(get_dep sphinxcontrib-bibtex $SPHINXCONTRIB_BIBTEX_VERSION)" \
    "$(get_dep sphinx-copybutton $SPHINXCONTRIB_BIBTEX_VERSION)" \
    "$(get_dep pydata-sphinx-theme $PYDATA_SPHINX_THEME_VERSION)" \
    "$(get_dep sphinx-design $SPHINX_DESIGN_VERSION)" \
    memory_profiler packaging seaborn pytest coverage compilers tensorflow

source activate $CONDA_ENV_NAME

# Build and install imbalanced-learn in dev mode
ls -l
pip install -e . --no-build-isolation

# The pipefail is requested to propagate exit code
set -o pipefail && cd doc && make $make_args 2>&1 | tee ~/log.txt

cd -
set +o pipefail
