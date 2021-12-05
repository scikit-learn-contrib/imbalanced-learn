#!/bin/bash

set -e

if [[ "$TEST_DOCSTRINGS" == 'true' ]]; then
    make test-doc
    pytest -vsl maint_tools/test_docstring.py
fi
