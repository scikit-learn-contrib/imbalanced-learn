import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('tree', parent_package, top_path)
    libraries = []
    config.add_extension('criterion',
                         sources=['criterion.c'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries)
                         # extra_compile_args=["-O3", "-fopenmp"],
                         # extra_link_args=["-fopenmp"])
    # config.add_subpackage("tests")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())