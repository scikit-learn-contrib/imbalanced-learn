""" Module to give helpful messages to the user that did not
compile package properly,

This code was adapted from scikit-learn's check_build utility.
"""
import os

PACKAGE_NAME = 'cython_template'

INPLACE_MSG = """
It appears that you are importing {package} from within the source tree.
Please either use an inplace install or try from another location.
""".format(package=PACKAGE_NAME)

STANDARD_MSG = """
If you have used an installer, please check that it is suited for your
Python version, your operating system and your platform.
"""

ERROR_TEMPLATE = """{error}
___________________________________________________________________________
Contents of {local_dir}:
{contents}
___________________________________________________________________________
It seems that the {package} has not been built correctly.

If you have installed {package} from source, please do not forget
to build the package before using it: run `python setup.py install`
in the source directory.
{msg}"""


def raise_build_error(e):
    # Raise a comprehensible error and list the contents of the
    # directory to help debugging on the mailing list.
    local_dir = os.path.split(__file__)[0]
    msg = STANDARD_MSG
    if local_dir == "megaman/__check_build":
        # Picking up the local install: this will work only if the
        # install is an 'inplace build'
        msg = INPLACE_MSG
    dir_content = list()
    for i, filename in enumerate(os.listdir(local_dir)):
        if ((i + 1) % 3):
            dir_content.append(filename.ljust(26))
        else:
            dir_content.append(filename + '\n')
    contents = ''.join(dir_content).strip()
    raise ImportError(ERROR_TEMPLATE.format(error=e,
                                            local_dir=local_dir,
                                            contents=contents,
                                            package=PACKAGE_NAME,
                                            msg=msg))

try:
    from ._check_build import check_build
except ImportError as e:
    raise_build_error(e)
