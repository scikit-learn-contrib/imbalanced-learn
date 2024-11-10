from dataclasses import dataclass

import sklearn
from sklearn.utils.fixes import parse_version

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)

if sklearn_version >= parse_version("1.6"):
    from sklearn.utils._tags import InputTags

    @dataclass
    class InputTags(InputTags):
        dataframe: bool = True
