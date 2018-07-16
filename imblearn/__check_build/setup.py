# Adapted from scikit-learn __check_build script (BSD-licensed)

import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('__check_build', parent_package, top_path)
    config.add_extension('_check_build',
                         sources=['_check_build.c'])
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
