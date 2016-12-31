.PHONY: all clean test
PYTHON=python
NOSETESTS=nosetests

all:
	$(PYTHON) setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" -o -name "*~" | xargs rm -f
	find . -name "*.pyx" -exec ./tools/rm_pyx_c_file.sh {} \;
	rm -rf coverage
	rm -rf dist
	rm -rf build
	rm -rf doc/auto_examples
	rm -rf doc/generated
	rm -rf doc/modules
	rm -rf examples/.ipynb_checkpoints

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code:
	$(NOSETESTS) -s -v imblearn

test-doc:
ifeq ($(BITS),64)
	$(NOSETESTS) -s -v doc/*.rst
endif

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) imblearn -s -v --with-coverage --cover-package=imblearn

test: test-coverage test-doc

html:
	conda install -y sphinx sphinx_rtd_theme numpydoc
	export SPHINXOPTS=-W; make -C doc html

conda:
	conda-build conda-recipe

code-analysis:
	flake8 imblearn | grep -v __init__
	pylint -E imblearn/ -d E1103,E0611,E1101

flake8-diff:
	./build_tools/travis/flake8_diff.sh
