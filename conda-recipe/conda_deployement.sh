#!/bin/bash

# This script is used to build and upload package to conda cloud
# in an automatic manner
mkdir build_conda
conda build build_conda imbalanced-learn
dirs=`ls -l "$PWD/build_conda"`
for d in build_conda/*/ ; do
    for file in $d*
    do
        if [[ -f $file ]]; then
            # upload each package
            anaconda upload $file
        fi
    done
done
rm -r build_conda
