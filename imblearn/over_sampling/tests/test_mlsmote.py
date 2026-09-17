"""Test the MLSMOTE multilabel over-sampler."""
# Authors: Omar Alaaeldein
# License: MIT

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import assert_allclose, assert_array_equal

from imblearn.over_sampling import MLSMOTE

RND_SEED = 0


@pytest.fixture
def multilabel_data():
    return make_multilabel_classification(
        n_samples=60, n_classes=4, n_labels=2, random_state=3
    )


def test_mlsmote_init():
    sampler = MLSMOTE(random_state=RND_SEED, k_neighbors=3, labelset_strategy="union")
    assert sampler.sampling_strategy == "auto"
    assert sampler.random_state == RND_SEED
    assert sampler.k_neighbors == 3
    assert sampler.categorical_features is None
    assert sampler.labelset_strategy == "union"


def test_mlsmote_minority_bags():
    # counts: label 0 -> 3, label 1 -> 2; IRLbl = [1, 1.5]; MeanIR = 1.25
    y = np.array([[1, 0], [1, 0], [1, 1], [0, 1]])
    X = np.random.RandomState(0).randn(4, 2)
    sampler = MLSMOTE(random_state=RND_SEED)
    bags = sampler._minority_bags(y)
    assert list(bags) == [1]
    assert_array_equal(bags[1], [2, 3])
    sampler.fit(X, y)
    assert sampler.sampling_strategy_ == {0: 0, 1: 2}


def test_mlsmote_pairwise_distances():
    # Hand-computed Euclidean + Value Difference Metric distances.
    X = np.array(
        [
            [0.0, "a"],
            [1.0, "a"],
            [3.0, "b"],
            [5.0, "a"],
            [6.0, "b"],
            [7.0, "b"],
        ],
        dtype=object,
    )
    y = np.array(
        [[1, 0], [1, 0], [1, 1], [0, 1], [0, 0], [0, 1]],
    )
    sampler = MLSMOTE(random_state=RND_SEED, categorical_features=[1])
    X_checked, y_checked, _ = sampler._check_X_y(X, y)
    categorical, continuous = sampler._resolve_feature_types(2)
    X_continuous = np.asarray(X_checked[:, continuous], dtype=np.float64)
    X_categorical = X_checked[:, categorical]
    from imblearn.over_sampling._mlsmote import _positive_proba_per_value

    proba_per_feature = [
        _positive_proba_per_value(X_categorical[:, 0], y_checked[:, 0])
    ]
    bag = np.array([0, 1, 2])
    distances = sampler._pairwise_distances(
        X_continuous, X_categorical, proba_per_feature, bag
    )
    # P(y0=1 | 'a') = 2/3, P(y0=1 | 'b') = 1/3, hence a VDM of 2/3 between
    # 'a' and 'b' and 0 between identical values.
    expected = np.array(
        [
            [np.inf, 1.0, np.sqrt(9.0 + 2.0 / 3.0)],
            [1.0, np.inf, np.sqrt(4.0 + 2.0 / 3.0)],
            [np.sqrt(9.0 + 2.0 / 3.0), np.sqrt(4.0 + 2.0 / 3.0), np.inf],
        ]
    )
    assert_allclose(distances, expected)


def test_mlsmote_fit_resample_generates_one_sample_per_seed(multilabel_data):
    X, y = multilabel_data
    sampler = MLSMOTE(random_state=RND_SEED, k_neighbors=3)
    X_res, y_res = sampler.fit_resample(X, y)
    # a single minority label with 17 samples: 60 + 17 synthetic samples
    assert X_res.shape == (77, X.shape[1])
    assert y_res.shape == (77, y.shape[1])
    assert_array_equal(X_res[:60], X)
    assert_array_equal(y_res[:60], y)
    assert sampler.sampling_strategy_ == {0: 0, 1: 0, 2: 17, 3: 0}
    assert np.isin(y_res, [0, 1]).all()


def test_mlsmote_deterministic(multilabel_data):
    X, y = multilabel_data
    X_first, y_first = MLSMOTE(random_state=1).fit_resample(X, y)
    X_second, y_second = MLSMOTE(random_state=1).fit_resample(X, y)
    assert_array_equal(X_first, X_second)
    assert_array_equal(y_first, y_second)


def test_mlsmote_labelset_strategies_nested(multilabel_data):
    # With a fixed seed, all strategies select the same neighbors and
    # reference samples: only the labelsets differ and they are nested.
    X, y = multilabel_data
    outputs = {
        strategy: MLSMOTE(
            random_state=RND_SEED,
            k_neighbors=3,
            labelset_strategy=strategy,
        ).fit_resample(X, y)
        for strategy in ("ranking", "union", "intersection")
    }
    shapes = {X_res.shape for X_res, _ in outputs.values()}
    assert shapes == {(77, X.shape[1])}
    for strategy in ("union", "intersection"):
        assert_array_equal(outputs[strategy][0], outputs["ranking"][0])
    n_samples = X.shape[0]
    ranking = outputs["ranking"][1][n_samples:]
    union = outputs["union"][1][n_samples:]
    intersection = outputs["intersection"][1][n_samples:]
    assert ((intersection <= ranking) & (ranking <= union)).all()


def test_mlsmote_categorical_features(multilabel_data):
    X, y = multilabel_data
    rng = np.random.RandomState(RND_SEED)
    X_mixed = np.column_stack(
        [rng.randint(0, 3, size=X.shape[0]).astype(np.float64), X[:, :4]]
    )
    sampler = MLSMOTE(random_state=RND_SEED, k_neighbors=3, categorical_features=[0])
    X_res, y_res = sampler.fit_resample(X_mixed, y)
    assert X_res.shape[0] == y_res.shape[0] > X.shape[0]
    assert set(np.unique(X_res[:, 0])) <= {0.0, 1.0, 2.0}


def test_mlsmote_categorical_features_bool_mask(multilabel_data):
    X, y = multilabel_data
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[0] = True
    X_res, _ = MLSMOTE(random_state=RND_SEED, categorical_features=mask).fit_resample(
        X, y
    )
    assert X_res.shape[0] > X.shape[0]


def test_mlsmote_string_categorical_features():
    pytest.importorskip("pandas")
    import pandas as pd

    X, y = make_multilabel_classification(
        n_samples=40, n_classes=3, n_labels=2, random_state=0
    )
    columns = [f"f{i}" for i in range(X.shape[1])]
    X_frame = pd.DataFrame(X, columns=columns)
    X_frame["color"] = np.where(np.arange(X.shape[0]) % 2, "red", "blue")
    y_frame = pd.DataFrame(y, columns=[f"l{i}" for i in range(y.shape[1])])
    sampler = MLSMOTE(
        random_state=RND_SEED, categorical_features=[X_frame.shape[1] - 1]
    )
    X_res, y_res = sampler.fit_resample(X_frame, y_frame)
    assert isinstance(X_res, pd.DataFrame)
    assert isinstance(y_res, pd.DataFrame)
    assert list(X_res.columns) == list(X_frame.columns)
    assert list(y_res.columns) == list(y_frame.columns)
    assert set(X_res["color"].unique()) <= {"red", "blue"}


def test_mlsmote_dataframe_roundtrip(multilabel_data):
    pytest.importorskip("pandas")
    import pandas as pd

    X, y = multilabel_data
    X_frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_frame = pd.DataFrame(y, columns=[f"l{i}" for i in range(y.shape[1])])
    X_res, y_res = MLSMOTE(random_state=RND_SEED).fit_resample(X_frame, y_frame)
    assert isinstance(X_res, pd.DataFrame)
    assert isinstance(y_res, pd.DataFrame)
    assert list(X_res.columns) == list(X_frame.columns)
    assert list(y_res.columns) == list(y_frame.columns)


def test_mlsmote_balanced_data_is_noop():
    rng = np.random.RandomState(RND_SEED)
    X = rng.randn(30, 4)
    y = np.tile(np.eye(3, dtype=np.int64), (10, 1))
    X_res, y_res = MLSMOTE(random_state=RND_SEED).fit_resample(X, y)
    assert X_res.shape == X.shape
    assert_array_equal(X_res, X)
    assert_array_equal(y_res, y)


def test_mlsmote_single_sample_label_is_skipped():
    X, y = make_multilabel_classification(
        n_samples=40, n_classes=3, n_labels=2, random_state=0
    )
    y[:, 0] = 0
    y[0, 0] = 1
    X_res, y_res = MLSMOTE(random_state=RND_SEED).fit_resample(X, y)
    assert X_res.shape[0] >= X.shape[0]
    assert y_res.shape[0] == X_res.shape[0]


def test_mlsmote_k_neighbors_larger_than_bag(multilabel_data):
    X, y = multilabel_data
    X_res, y_res = MLSMOTE(random_state=RND_SEED, k_neighbors=100).fit_resample(X, y)
    assert X_res.shape == (77, X.shape[1])
    assert y_res.shape == (77, y.shape[1])


def test_mlsmote_rejects_single_label_target(multilabel_data):
    X, _ = multilabel_data
    y = np.array([0] * 40 + [1] * 20)
    with pytest.raises(ValueError, match="multilabel-indicator"):
        MLSMOTE().fit_resample(X, y)


def test_mlsmote_rejects_continuous_target(multilabel_data):
    X, _ = multilabel_data
    with pytest.raises(ValueError, match="Unknown label type"):
        MLSMOTE().fit_resample(X, np.random.rand(X.shape[0], 2))


def test_mlsmote_rejects_label_without_positive_sample(multilabel_data):
    X, y = multilabel_data
    y = y.copy()
    y[:, 0] = 0
    with pytest.raises(ValueError, match="at least one positive sample"):
        MLSMOTE().fit_resample(X, y)


def test_mlsmote_rejects_sparse_input(multilabel_data):
    X, y = multilabel_data
    with pytest.raises(ValueError, match="does not support sparse"):
        MLSMOTE().fit_resample(sparse.csr_matrix(X), y)


def test_mlsmote_rejects_non_numeric_continuous(multilabel_data):
    X, y = multilabel_data
    X = X.astype(object)
    X[0, 0] = "not-a-number"
    with pytest.raises(ValueError, match="continuous features to be numeric"):
        MLSMOTE().fit_resample(X, y)


def test_mlsmote_invalid_categorical_features(multilabel_data):
    X, y = multilabel_data
    with pytest.raises(ValueError, match="Categorical feature indices"):
        MLSMOTE(categorical_features=[X.shape[1]]).fit_resample(X, y)
    with pytest.raises(ValueError, match="boolean mask"):
        MLSMOTE(categorical_features=np.zeros(2, dtype=bool)).fit_resample(X, y)


def test_mlsmote_invalid_params(multilabel_data):
    from sklearn.utils._param_validation import InvalidParameterError

    X, y = multilabel_data
    with pytest.raises(InvalidParameterError, match="sampling_strategy"):
        MLSMOTE(sampling_strategy="minority").fit(X, y)
    with pytest.raises(InvalidParameterError, match="labelset_strategy"):
        MLSMOTE(labelset_strategy="unknown").fit(X, y)
    with pytest.raises(InvalidParameterError, match="k_neighbors"):
        MLSMOTE(k_neighbors=0).fit(X, y)


def test_mlsmote_not_fitted(multilabel_data):
    sampler = MLSMOTE()
    with pytest.raises(NotFittedError):
        sampler.__sklearn_tags__() and sampler.get_feature_names_out()
