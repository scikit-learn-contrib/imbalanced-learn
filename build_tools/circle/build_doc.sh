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

MAKE_TARGET=html

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
    deactivate
fi

# Install dependencies with miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
export PATH="$MINICONDA_PATH/bin:$PATH"
conda update --yes --quiet conda

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n $CONDA_ENV_NAME --yes --quiet python=3
source activate $CONDA_ENV_NAME

conda install --yes pip numpy scipy scikit-learn pillow matplotlib sphinx \
      sphinx_rtd_theme numpydoc
pip install -U git+https://github.com/sphinx-gallery/sphinx-gallery.git

# Build and install imbalanced-learn in dev mode
ls -l
pip install -e .

# The pipefail is requested to propagate exit code
set -o pipefail && cd doc && make $MAKE_TARGET 2>&1 | tee ~/log.txt

cd -
set +o pipefail
