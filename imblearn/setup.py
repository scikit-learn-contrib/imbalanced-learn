def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('imblearn', parent_package, top_path)

    config.add_subpackage('combine')
    config.add_subpackage('combine/tests')
    config.add_subpackage('ensemble')
    config.add_subpackage('ensemble/tests')
    config.add_subpackage('metrics')
    config.add_subpackage('metrics/tests')
    config.add_subpackage('over_sampling')
    config.add_subpackage('over_sampling/tests')
    config.add_subpackage('under_sampling')
    config.add_subpackage('under_sampling/tests')

    config.add_subpackage('tests')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    config = configuration(top_path='').todict()
    setup(**config)
