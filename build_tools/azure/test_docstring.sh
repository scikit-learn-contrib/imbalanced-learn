#!/bin/bash

set -e

if [[ "$TEST_DOCSTRINGS" == 'true' ]]; then
    pytest -vsl maint_tools/test_docstring.py
fi
