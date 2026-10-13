"""Tests for the cleaning and labeling logic of the commit message example.

The functions under test live in the imbalanced-learn example
``examples/applications/plot_commit_message_classification.py``. The example
is executed once per test session to load them. Set ``COMMIT_EXAMPLE_PATH``
to point at the example if the repository is not cloned next to this folder.
"""

import os
import runpy
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent
    / "imbalanced-learn"
    / "examples"
    / "applications"
    / "plot_commit_message_classification.py"
)
EXAMPLE_PATH = Path(os.environ.get("COMMIT_EXAMPLE_PATH", DEFAULT_PATH))


@pytest.fixture(scope="session")
def example():
    if not EXAMPLE_PATH.exists():
        pytest.fail(f"Example not found at {EXAMPLE_PATH}")
    with warnings.catch_warnings():
        # plt.show() warns with the non-interactive Agg backend
        warnings.filterwarnings("ignore", message=".*non-interactive")
        return runpy.run_path(str(EXAMPLE_PATH))


@pytest.fixture(scope="session")
def clean_message(example):
    return example["clean_message"]


@pytest.fixture(scope="session")
def label_commit(example):
    return example["label_commit"]


# clean_message ---------------------------------------------------------------


@pytest.mark.parametrize("message", [None, "", "   ", "\n\n"])
def test_clean_message_empty_input(clean_message, message):
    assert clean_message(message) == ""


def test_clean_message_lowercases(clean_message):
    assert clean_message("Fix Memory Leak In CACHE") == "fix memory leak in cache"


def test_clean_message_keeps_subject_line_only(clean_message):
    message = "Update parser\n\nThis fixes a bug.\nSecond paragraph."
    assert clean_message(message) == "update parser"


def test_clean_message_skips_leading_blank_lines(clean_message):
    assert clean_message("\n\n  Add retry policy\nbody") == "add retry policy"


def test_clean_message_handles_windows_line_endings(clean_message):
    assert clean_message("Refactor cli\r\n\r\nbody") == "refactor cli"


@pytest.mark.parametrize(
    "message",
    [
        "Fix link https://example.com/a?b=1 in docs",
        "Fix link http://example.com in docs",
        "Fix link www.example.com/page in docs",
    ],
)
def test_clean_message_strips_urls(clean_message, message):
    assert clean_message(message) == "fix link in docs"


@pytest.mark.parametrize(
    "message",
    ["Fix timeout (#123)", "Fix timeout #123", "Fix #123 timeout", "Fix timeout(#1)"],
)
def test_clean_message_strips_issue_refs(clean_message, message):
    assert clean_message(message) == "fix timeout"


def test_clean_message_keeps_hash_without_number(clean_message):
    assert clean_message("Support C# files") == "support c# files"


def test_clean_message_collapses_whitespace(clean_message):
    assert clean_message("  Fix   race\tcondition  ") == "fix race condition"


def test_clean_message_url_in_body_is_dropped_with_body(clean_message):
    message = "Update docs\n\nSee https://example.com/docs/1"
    assert clean_message(message) == "update docs"


# label_commit ----------------------------------------------------------------


@pytest.mark.parametrize("message", [None, "", "   "])
def test_label_commit_empty_input_is_other(label_commit, message):
    assert label_commit(message) == "other"


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Fix memory leak in cache", "fix"),
        ("Bugfix: timeout in cli", "fix"),
        ("Add retry policy to http client", "feature"),
        ("Implement plugin hooks", "feature"),
        ("Update changelog", "other"),
        ("Bump version to 0.3.0", "other"),
    ],
)
def test_label_commit_keywords(label_commit, message, expected):
    assert label_commit(message) == expected


def test_label_commit_add_and_bug_is_fix(label_commit):
    # "fix" keywords take precedence over "feature" keywords
    assert label_commit("Add regression test for bug #12") == "fix"
    assert label_commit("add check for the bug in parser") == "fix"


def test_label_commit_is_case_insensitive(label_commit):
    assert label_commit("FIX CRASH ON STARTUP") == "fix"
    assert label_commit("ADD option") == "feature"


def test_label_commit_reads_the_body(label_commit):
    # Unlike clean_message, the label uses the whole message.
    assert label_commit("Update parser\n\nFixes #123.") == "fix"
    assert label_commit("Rework cli internals\n\nThis adds a dry-run mode.") == (
        "feature"
    )
    assert label_commit("Update cache\n\nNo functional change.") == "other"


def test_label_commit_bug_in_body_overrides_feature_subject(label_commit):
    assert label_commit("Add caching layer\n\nAlso fixes a crash.") == "fix"


@pytest.mark.parametrize(
    "message",
    ["Update address book", "Rename prefix handling", "Debugging notes", "Addendum"],
)
def test_label_commit_matches_whole_words_only(label_commit, message):
    # "address" contains "add", "prefix" contains "fix", "debugging" contains
    # "bug": none of them is a keyword.
    assert label_commit(message) == "other"


def test_label_commit_conventional_commit_prefix(label_commit):
    assert label_commit("fix: handle empty input") == "fix"
    assert label_commit("feat(cli): add --verbose") == "feature"


# generated dataset -----------------------------------------------------------


def test_generated_messages_are_reproducible(example):
    make = example["make_commit_messages"]
    weights = {"fix": 0.7, "feature": 0.2, "other": 0.1}
    assert make(50, weights, random_state=1) == make(50, weights, random_state=1)


def test_generated_class_proportions(example):
    counts = Counter(example["y"].tolist())
    total = sum(counts.values())
    assert counts["fix"] / total == pytest.approx(0.7, abs=0.05)
    assert counts["feature"] / total == pytest.approx(0.2, abs=0.05)
    assert counts["other"] / total == pytest.approx(0.1, abs=0.05)


def test_cleaned_messages_have_no_issue_refs_or_urls(example):
    for text in example["X"]:
        assert "#" not in text
        # "http client" is a module name, so look for URL syntax instead
        assert "://" not in text
        assert "www." not in text
        assert text == text.lower()
