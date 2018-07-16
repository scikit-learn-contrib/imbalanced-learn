import os

PACKAGE_NAME = 'imblearn/tree_split'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(PACKAGE_NAME, parent_package, top_path)
    
    config.add_extension('hellinger_distance_criterion',
                         sources=['hellinger_distance_criterion.c'])
    config.li

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
