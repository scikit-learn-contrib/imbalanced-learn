.PHONY: all clean test

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" -o -name "*~" | xargs rm -f
	find . -name "*.pyx" -exec ./tools/rm_pyx_c_file.sh {} \;
	rm -rf coverage
	rm -rf dist
	rm -rf build
	rm -rf doc/_build
	rm -rf doc/auto_examples
	rm -rf doc/generated
	rm -rf doc/modules
	rm -rf examples/.ipynb_checkpoints

test-code:
	pytest imblearn

test-doc:
	pytest doc/*.rst

test-coverage:
	rm -rf coverage .coverage
	pytest --cov=imblearn imblearn

test: test-coverage test-doc

html:
	export SPHINXOPTS=-W; make -C doc html

conda:
	conda-build conda-recipe

code-analysis:
	flake8 imblearn | grep -v __init__
	pylint -E imblearn/ -d E1103,E0611,E1101
