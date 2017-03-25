sudo -E apt-get -yq remove texlive-binaries --purge
sudo apt-get update
sudo apt-get install libatlas-dev libatlas3gf-base
sudo apt-get install build-essential python-dev python-setuptools
# install numpy first as it is a compile time dependency for other packages
pip install --upgrade numpy
pip install --upgrade scipy matplotlib setuptools nose coverage sphinx pillow sphinx_rtd_theme
# Installing required packages for `make -C doc check command` to work.
sudo -E apt-get -yq update
sudo -E apt-get -yq --no-install-suggests --no-install-recommends --force-yes install dvipng texlive-latex-base texlive-latex-extra
pip install --upgrade cython numpydoc
pip install --upgrade scikit-learn
pip install --upgrade seaborn
git clone git@github.com:glemaitre/sphinx-gallery.git
cd sphinx-gallery
pip install .
cd ../
