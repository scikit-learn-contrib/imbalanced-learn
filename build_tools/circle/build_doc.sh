#!/usr/bin/env bash
set -x
set -e

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
    deactivate
fi

# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash
export PATH=/home/circleci/.pixi/bin:$PATH

# The pipefail is requested to propagate exit code
set -o pipefail && pixi run --frozen -e docs build-docs 2>&1 | tee ~/log.txt
set +o pipefail
