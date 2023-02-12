import importlib
import inspect
import pkgutil
import re
from inspect import signature
from typing import Optional

import pytest

import imblearn
from imblearn.utils.testing import all_estimators

numpydoc_validation = pytest.importorskip("numpydoc.validate")

# List of whitelisted modules and methods; regexp are supported.
# These docstrings will fail because they are inheriting from scikit-learn
DOCSTRING_WHITELIST = [
    "ADASYN$",
    "ADASYN.",
    "AllKNN$",
    "AllKNN.",
    "BalancedBaggingClassifier$",
    "BalancedBaggingClassifier.",
    "BalancedRandomForestClassifier$",
    "BalancedRandomForestClassifier.",
    "ClusterCentroids$",
    "ClusterCentroids.",
    "CondensedNearestNeighbour$",
    "CondensedNearestNeighbour.",
    "EasyEnsembleClassifier$",
    "EasyEnsembleClassifier.",
    "EditedNearestNeighbours$",
    "EditedNearestNeighbours.",
    "FunctionSampler$",
    "FunctionSampler.",
    "InstanceHardnessThreshold$",
    "InstanceHardnessThreshold.",
    "SMOTE$",
    "SMOTE.",
    "NearMiss$",
    "NearMiss.",
    "NeighbourhoodCleaningRule$",
    "NeighbourhoodCleaningRule.",
    "OneSidedSelection$",
    "OneSidedSelection.",
    "Pipeline$",
    "Pipeline.",
    "RUSBoostClassifier$",
    "RUSBoostClassifier.",
    "RandomOverSampler$",
    "RandomOverSampler.",
    "RandomUnderSampler$",
    "RandomUnderSampler.",
    "TomekLinks$",
    "TomekLinks",
    "ValueDifferenceMetric$",
    "ValueDifferenceMetric.",
]

FUNCTION_DOCSTRING_IGNORE_LIST = [
    "imblearn.tensorflow._generator.balanced_batch_generator",
]
FUNCTION_DOCSTRING_IGNORE_LIST = set(FUNCTION_DOCSTRING_IGNORE_LIST)


def get_all_methods():
    estimators = all_estimators()
    for name, Estimator in estimators:
        if name.startswith("_"):
            # skip private classes
            continue
        methods = []
        for name in dir(Estimator):
            if name.startswith("_"):
                continue
            method_obj = getattr(Estimator, name)
            if hasattr(method_obj, "__call__") or isinstance(method_obj, property):
                methods.append(name)
        methods.append(None)

        for method in sorted(methods, key=lambda x: str(x)):
            yield Estimator, method


def _is_checked_function(item):
    if not inspect.isfunction(item):
        return False

    if item.__name__.startswith("_"):
        return False

    mod = item.__module__
    if not mod.startswith("imblearn.") or mod.endswith("estimator_checks"):
        return False

    return True


def get_all_functions_names():
    """Get all public functions define in the imblearn module"""
    modules_to_ignore = {
        "tests",
        "estimator_checks",
    }

    all_functions_names = set()
    for module_finder, module_name, ispkg in pkgutil.walk_packages(
        path=imblearn.__path__, prefix="imblearn."
    ):
        module_parts = module_name.split(".")
        if (
            any(part in modules_to_ignore for part in module_parts)
            or "._" in module_name
        ):
            continue

        module = importlib.import_module(module_name)
        functions = inspect.getmembers(module, _is_checked_function)
        for name, func in functions:
            full_name = f"{func.__module__}.{func.__name__}"
            all_functions_names.add(full_name)

    return sorted(all_functions_names)


def filter_errors(errors, method, Estimator=None):
    """
    Ignore some errors based on the method type.

    These rules are specific for scikit-learn."""
    for code, message in errors:
        # We ignore following error code,
        #  - RT02: The first line of the Returns section
        #    should contain only the type, ..
        #   (as we may need refer to the name of the returned
        #    object)
        #  - GL01: Docstring text (summary) should start in the line
        #    immediately after the opening quotes (not in the same line,
        #    or leaving a blank line in between)
        #  - GL02: If there's a blank line, it should be before the
        #    first line of the Returns section, not after (it allows to have
        #    short docstrings for properties).

        if code in ["RT02", "GL01", "GL02"]:
            continue

        # Ignore PR02: Unknown parameters for properties. We sometimes use
        # properties for ducktyping, i.e. SGDClassifier.predict_proba
        if code == "PR02" and Estimator is not None and method is not None:
            method_obj = getattr(Estimator, method)
            if isinstance(method_obj, property):
                continue

        # Following codes are only taken into account for the
        # top level class docstrings:
        #  - ES01: No extended summary found
        #  - SA01: See Also section not found
        #  - EX01: No examples section found

        if method is not None and code in ["EX01", "SA01", "ES01"]:
            continue
        yield code, message


def repr_errors(res, estimator=None, method: Optional[str] = None) -> str:
    """Pretty print original docstring and the obtained errors

    Parameters
    ----------
    res : dict
        result of numpydoc.validate.validate
    estimator : {estimator, None}
        estimator object or None
    method : str
        if estimator is not None, either the method name or None.

    Returns
    -------
    str
       String representation of the error.
    """
    if method is None:
        if hasattr(estimator, "__init__"):
            method = "__init__"
        elif estimator is None:
            raise ValueError("At least one of estimator, method should be provided")
        else:
            raise NotImplementedError

    if estimator is not None:
        obj = getattr(estimator, method)
        try:
            obj_signature = signature(obj)
        except TypeError:
            # In particular we can't parse the signature of properties
            obj_signature = (
                "\nParsing of the method signature failed, "
                "possibly because this is a property."
            )

        obj_name = estimator.__name__ + "." + method
    else:
        obj_signature = ""
        obj_name = method

    msg = "\n\n" + "\n\n".join(
        [
            str(res["file"]),
            obj_name + str(obj_signature),
            res["docstring"],
            "# Errors",
            "\n".join(
                " - {}: {}".format(code, message) for code, message in res["errors"]
            ),
        ]
    )
    return msg


@pytest.mark.parametrize("function_name", get_all_functions_names())
def test_function_docstring(function_name, request):
    """Check function docstrings using numpydoc."""
    if function_name in FUNCTION_DOCSTRING_IGNORE_LIST:
        request.applymarker(
            pytest.mark.xfail(run=False, reason="TODO pass numpydoc validation")
        )

    res = numpydoc_validation.validate(function_name)

    res["errors"] = list(filter_errors(res["errors"], method="function"))

    if res["errors"]:
        msg = repr_errors(res, method=f"Tested function: {function_name}")

        raise ValueError(msg)


@pytest.mark.parametrize("Estimator, method", get_all_methods())
def test_docstring(Estimator, method, request):
    base_import_path = Estimator.__module__
    import_path = [base_import_path, Estimator.__name__]
    if method is not None:
        import_path.append(method)

    import_path = ".".join(import_path)

    if not any(re.search(regex, import_path) for regex in DOCSTRING_WHITELIST):
        request.applymarker(
            pytest.mark.xfail(run=False, reason="TODO pass numpydoc validation")
        )

    res = numpydoc_validation.validate(import_path)

    res["errors"] = list(filter_errors(res["errors"], method))

    if res["errors"]:
        msg = repr_errors(res, Estimator, method)

        raise ValueError(msg)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate docstring with numpydoc.")
    parser.add_argument("import_path", help="Import path to validate")

    args = parser.parse_args()

    res = numpydoc_validation.validate(args.import_path)

    import_path_sections = args.import_path.split(".")
    # When applied to classes, detect class method. For functions
    # method = None.
    # TODO: this detection can be improved. Currently we assume that we have
    # class # methods if the second path element before last is in camel case.
    if len(import_path_sections) >= 2 and re.match(
        r"(?:[A-Z][a-z]*)+", import_path_sections[-2]
    ):
        method = import_path_sections[-1]
    else:
        method = None

    res["errors"] = list(filter_errors(res["errors"], method))

    if res["errors"]:
        msg = repr_errors(res, method=args.import_path)

        print(msg)
        sys.exit(1)
    else:
        print("All docstring checks passed for {}!".format(args.import_path))
