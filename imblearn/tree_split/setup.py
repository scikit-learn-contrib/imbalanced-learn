from setuptools   import setup
from Cython.Build import cythonize
import numpy

setup(
	name='hellinger-distance-criterion',
	version=0.1,
	url='github.com/EvgeniDubov/hellinger-random-forest',
    author='Evgeni Dubov',
	author_email='evgeni.dubov@gmail.com',
	ext_modules = cythonize('hellinger_distance_criterion.pyx'),
	include_dirs=[numpy.get_include()]
)