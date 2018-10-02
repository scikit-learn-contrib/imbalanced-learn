PACKAGE_NAME = 'imblearn'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(PACKAGE_NAME, parent_package, top_path)

    config.add_subpackage('__check_build')

    # pure python packages
    config.add_subpackage('combine')
    config.add_subpackage('combine/tests')
    config.add_subpackage('datasets')
    config.add_subpackage('datasets/tests')
    config.add_subpackage('ensemble')
    config.add_subpackage('ensemble/tests')
    config.add_subpackage('keras')
    config.add_subpackage('keras/tests')
    config.add_subpackage('metrics')
    config.add_subpackage('metrics/tests')
    config.add_subpackage('tensorflow')
    config.add_subpackage('tensorflow/tests')
    config.add_subpackage('tests')
    config.add_subpackage('under_sampling')
    config.add_subpackage('under_sampling/_prototype_generation')
    config.add_subpackage('under_sampling/_prototype_generation/tests')
    config.add_subpackage('under_sampling/_prototype_selection')
    config.add_subpackage('under_sampling/_prototype_selection/tests')
    config.add_subpackage('utils')
    config.add_subpackage('utils/tests')

    # packages that have their own setup.py -> cython files
    config.add_subpackage('tree')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
