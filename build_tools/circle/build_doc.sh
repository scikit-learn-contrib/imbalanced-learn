#!/usr/bin/env bash
set -x
set -e

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
    deactivate
fi

# Install pixi
# Keep this version in sync with `pixi-version` in `.github/workflows/*.yml`.
PIXI_VERSION="v0.70.1"
curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION="${PIXI_VERSION}" bash
export PATH=/home/circleci/.pixi/bin:$PATH

# The pipefail is requested to propagate exit code
set -o pipefail && pixi run --frozen -e docs build-docs 2>&1 | tee ~/log.txt
set +o pipefail
