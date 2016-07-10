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

test:
	$(NOSETESTS) -s -v imblearn

coverage:
	$(NOSETESTS) imblearn -s -v --with-coverage --cover-package=imblearn

html:
	conda install -y sphinx sphinx_rtd_theme numpydoc
	export SPHINXOPTS=-W; make -C doc html

conda:
	conda-build conda-recipe
