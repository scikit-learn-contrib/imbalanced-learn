"""Tests for the helper functions of ``plot_sentiment_svm_resampling.py``.

Gallery examples run their whole experiment when they are executed, so the
example cannot simply be imported. Instead, we parse the file and execute only
the imports, the upper-case constants and the function definitions. The tests
therefore need no network access and run in well under a second.
"""

# Authors: Mansi Verma
# License: MIT

import ast
from pathlib import Path

import numpy as np
import pytest
from sklearn.svm import LinearSVC

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

EXAMPLE = Path(__file__).resolve().parents[1] / "plot_sentiment_svm_resampling.py"


def _load_definitions():
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            keep.append(node)
        elif isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id.isupper()
            for target in node.targets
        ):
            keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    namespace = {"__name__": "plot_sentiment_svm_resampling"}
    exec(compile(module, str(EXAMPLE), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def ex():
    return _load_definitions()


@pytest.fixture(scope="module")
def toy_corpus():
    negative = ["awful terrible", "terrible service awful", "bad awful day"]
    neutral = ["the meeting is at 5", "store opens at 9", "train leaves at 7 today"]
    positive = ["great love it", "love this great phone", "great day love"]
    texts = (negative + neutral + positive) * 4
    labels = np.array(([0] * 3 + [1] * 3 + [2] * 3) * 4)
    return texts, labels


def test_label_names(ex):
    assert list(ex["LABEL_NAMES"]) == ["negative", "neutral", "positive"]


def test_pipeline_without_sampler(ex):
    pipe = ex["make_sentiment_pipeline"]()
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["tfidfvectorizer", "linearsvc"]


def test_pipeline_with_sampler_and_weights(ex):
    sampler = RandomUnderSampler(random_state=0)
    pipe = ex["make_sentiment_pipeline"](
        sampler=sampler, class_weight="balanced", C=2.0
    )
    assert len(pipe.steps) == 3
    assert pipe.steps[1][1] is sampler
    svm = pipe.steps[-1][1]
    assert isinstance(svm, LinearSVC)
    assert svm.class_weight == "balanced"
    assert svm.C == 2.0


def test_strategies_are_independent_and_complete(ex):
    first, second = ex["make_strategies"](), ex["make_strategies"]()
    assert list(first) == [
        "No correction",
        "class_weight='balanced'",
        "Random under-sampling",
        "Random over-sampling",
    ]
    for name in first:
        assert first[name] is not second[name]


def test_sampler_is_not_applied_at_predict_time(ex, toy_corpus):
    texts, labels = toy_corpus
    texts, labels = texts[:-6], labels[:-6]  # make the positive class smaller
    pipe = ex["make_sentiment_pipeline"](sampler=RandomUnderSampler(random_state=0))
    pipe.fit(texts, labels)
    assert len(pipe.predict(texts)) == len(texts)


@pytest.mark.parametrize(
    "text, expected",
    [("awful terrible", 0), ("the meeting is at 5 today", 1), ("great love it", 2)],
)
def test_predicts_all_three_sentiments(ex, toy_corpus, text, expected):
    pipe = ex["make_sentiment_pipeline"](C=1.0).fit(*toy_corpus)
    assert pipe.predict([text])[0] == expected


def test_evaluate_returns_bounded_scores(ex, toy_corpus):
    texts, labels = toy_corpus
    scores = ex["evaluate"](
        ex["make_sentiment_pipeline"](), texts, labels, texts, labels
    )
    assert set(scores) == {
        "macro F1",
        "balanced accuracy",
        "geometric mean",
        "recall negative",
        "recall neutral",
        "recall positive",
    }
    assert all(0.0 <= value <= 1.0 for value in scores.values())


def _write_cache(folder, subset, texts, labels):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{subset}_text.txt").write_text("\n".join(texts), encoding="utf-8")
    (folder / f"{subset}_labels.txt").write_text(
        "\n".join(str(label) for label in labels), encoding="utf-8"
    )


def test_loader_reads_cached_files_without_network(ex, tmp_path, monkeypatch):
    _write_cache(tmp_path / "tweeteval_sentiment", "val", ["only tweet"], [2])

    def fail(*args, **kwargs):
        raise AssertionError("the loader must not use the network")

    monkeypatch.setitem(ex["fetch_tweeteval_sentiment"].__globals__, "urlopen", fail)
    texts, labels = ex["fetch_tweeteval_sentiment"]("val", data_home=tmp_path)
    assert texts == ["only tweet"]
    assert labels.shape == (1,)
    assert labels[0] == 2


def test_loader_rejects_unknown_subset(ex, tmp_path):
    with pytest.raises(ValueError, match="subset must be"):
        ex["fetch_tweeteval_sentiment"]("validation", data_home=tmp_path)


def test_loader_detects_mismatched_files(ex, tmp_path):
    _write_cache(tmp_path / "tweeteval_sentiment", "test", ["a", "b"], [0])
    with pytest.raises(ValueError, match="2 texts but 1 labels"):
        ex["fetch_tweeteval_sentiment"]("test", data_home=tmp_path)
