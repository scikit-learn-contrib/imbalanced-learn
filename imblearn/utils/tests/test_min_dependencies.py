"""Tests for the minimum dependencies in the README.rst file."""

import os
import platform
import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import parse

import imblearn


@pytest.mark.skipif(
    platform.system() == "Windows" or parse(platform.python_version()) < parse("3.11"),
    reason="This test is enough on unix system and requires Python >= 3.11",
)
def test_min_dependencies_readme():
    # local import to not import the file with Python < 3.11
    import tomllib

    # Test that the minimum dependencies in the README.rst file are
    # consistent with the minimum dependencies defined at the file:
    # pyproject.toml

    pyproject_path = Path(imblearn.__path__[0]).parents[0] / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    def process_requirements(requirements):
        result = {}
        for req in requirements:
            req = Requirement(req)
            for specifier in req.specifier:
                if specifier.operator == ">=":
                    result[req.name] = parse(specifier.version)
        return result

    min_dependencies = process_requirements(
        [f"python{pyproject_data['project']['requires-python']}"]
    )
    min_dependencies.update(
        process_requirements(pyproject_data["project"]["dependencies"])
    )

    markers = ["docs", "optional", "tensorflow", "keras", "tests"]
    for marker_name in markers:
        min_dependencies.update(
            process_requirements(
                pyproject_data["project"]["optional-dependencies"][marker_name]
            )
        )

    pattern = re.compile(
        r"(\.\. \|)"
        + r"(([A-Za-z]+\-?)+)"
        + r"(MinVersion\| replace::)"
        + r"( [0-9]+\.[0-9]+(\.[0-9]+)?)"
    )

    readme_path = Path(imblearn.__path__[0]).parents[0]
    readme_file = readme_path / "README.rst"

    if not os.path.exists(readme_file):
        # Skip the test if the README.rst file is not available.
        # For instance, when installing scikit-learn from wheels
        pytest.skip("The README.rst file is not available.")

    with readme_file.open("r") as f:
        for line in f:
            matched = pattern.match(line)

            if not matched:
                continue

            package, version = matched.group(2), matched.group(5)
            package = package.lower()
            if package == "scikitlearn":
                package = "scikit-learn"

            if package in min_dependencies:
                version = parse(version)
                min_version = min_dependencies[package]

                assert version == min_version, f"{package} has a mismatched version"
