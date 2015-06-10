from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'scipy',
    'scikit-learn',
    ]

setup(name='UnbalancedDataset',
      version='0.1',
      description='Python module with numerous re-sampling strategies to deal '
                  'with classification of data-sets with strong between class '
                  'imbalance.',
      classifiers=[
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.4",
          ],
      author='Fernando Nogueira, Guillaume Lemaitre',
      author_email='fmfnogueira@gmail.com, guillaume.lemaitre@udg.edu',
      url='https://github.com/fmfn/UnbalancedDataset',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      )
