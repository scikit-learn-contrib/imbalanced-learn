#!/bin/bash

# This script is used to build and upload package to conda cloud
# in an automatic manner
for pyv in {2.7,3.4,3.5,3.6} ; do
    mkdir build_conda
    conda build --python=$pyv --output-folder build_conda imbalanced-learn
    conda convert --platform all build_conda/*.tar.bz2 \
          -o build_conda/
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
done
