#!/bin/bash

set -e

if [[ "$DISTRIB" =~ ^conda.* ]]; then
    source activate $VIRTUALENV
elif [[ "$DISTRIB" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

if [[ "TEST_DOCSTRING" == 'true' ]]; then
    make test-doc
    pytest -vsl maint_tools/test_docstring.py
fi
