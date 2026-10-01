"""Polars input/output support for samplers and pipelines."""

# License: MIT

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import assert_allclose, assert_array_equal

from imblearn import FunctionSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC, RandomOverSampler
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, TomekLinks
from imblearn.utils._validation import ArraysTransformer


@pytest.fixture
def pl():
    return pytest.importorskip("polars")


@pytest.fixture
def classification_data(pl):
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_redundant=0,
        weights=[0.8, 0.2],
        random_state=42,
    )
    return pl.DataFrame(X, schema=["a", "b", "c", "d"], orient="row"), pl.Series(
        "target", y, dtype=pl.Int32
    )


@pytest.mark.parametrize(
    "sampler",
    [
        RandomOverSampler(random_state=0),
        RandomOverSampler(random_state=0, shrinkage=0.1),
        RandomUnderSampler(random_state=0),
        SMOTE(random_state=0),
        TomekLinks(),
        ClusterCentroids(random_state=0),
        SMOTEENN(random_state=0),
        SMOTETomek(random_state=0),
        FunctionSampler(),
    ],
)
@pytest.mark.parametrize("target_type", ["series", "dataframe", "numpy"])
def test_sampler_polars_numeric(pl, classification_data, sampler, target_type):
    X, y = classification_data
    if target_type == "dataframe":
        y = y.to_frame()
    elif target_type == "numpy":
        y = y.to_numpy()
    y_array = np.asarray(y).ravel()
    X_expected, y_expected = clone(sampler).fit_resample(X.to_numpy(), y_array)
    X_res, y_res = sampler.fit_resample(X, y)

    assert isinstance(X_res, pl.DataFrame)
    assert X_res.schema == X.schema
    assert_allclose(X_res.to_numpy(), X_expected)
    if target_type == "series":
        assert isinstance(y_res, pl.Series)
        assert y_res.name == "target"
        assert y_res.dtype == pl.Int32
    elif target_type == "dataframe":
        assert isinstance(y_res, pl.DataFrame)
        assert y_res.schema == y.schema
    else:
        assert isinstance(y_res, np.ndarray)
    assert_array_equal(np.asarray(y_res).ravel(), y_expected)
    assert_array_equal(sampler.feature_names_in_, X.columns)


@pytest.mark.parametrize(
    "x_type,y_type",
    [
        ("polars", "pandas"),
        ("pandas", "polars"),
        ("numpy", "polars"),
    ],
)
def test_sampler_mixed_containers(pl, classification_data, x_type, y_type):
    pd = pytest.importorskip("pandas")
    X, y = classification_data
    if x_type == "pandas":
        X = pd.DataFrame(X.to_numpy(), columns=X.columns, index=np.arange(100) + 5)
    elif x_type == "numpy":
        X = X.to_numpy()
    if y_type == "pandas":
        y = pd.Series(y.to_numpy().copy(), name="target", index=np.arange(100) + 5)
    X_res, y_res = RandomUnderSampler(random_state=0).fit_resample(X, y)
    assert isinstance(X_res, type(X))
    assert isinstance(y_res, type(y))
    assert len(X_res) == len(y_res)
    assert y_res.name == y.name


@pytest.mark.parametrize(
    "sampler",
    [
        RandomOverSampler(random_state=0),
        RandomUnderSampler(random_state=0),
    ],
)
def test_sampler_polars_heterogeneous(pl, sampler):
    X = pl.DataFrame(
        {
            "count": pl.Series([1, None, 3, 4, 5, 6], dtype=pl.Int16),
            "value": [np.nan, None, 0.3, 0.4, 0.5, 0.6],
            "category": pl.Series(["a", "b", "a", "b", "a", "a"], dtype=pl.Categorical),
            "flag": [True, False, True, False, True, True],
        }
    )
    y = pl.Series("target", [0, 0, 0, 0, 1, 1], dtype=pl.Int8)
    X_before = X.clone()
    y_before = y.clone()
    X_res, y_res = sampler.fit_resample(X, y)
    assert X_res.equals(X[sampler.sample_indices_])
    assert X_res.schema == X.schema
    assert y_res.equals(y[sampler.sample_indices_])
    assert X.equals(X_before)
    assert y.equals(y_before)


@pytest.mark.parametrize("categorical_dtype", ["string", "categorical", "enum"])
@pytest.mark.parametrize("categorical_features", [["category"], [1], "auto"])
def test_smotenc_polars(pl, categorical_dtype, categorical_features):
    dtype = {
        "string": pl.String,
        "categorical": pl.Categorical,
        "enum": pl.Enum(["a", "b"]),
    }[categorical_dtype]
    X = pl.DataFrame(
        {
            "value": [0.1, 0.3, 0.2, 0.5, 0.6, 0.7],
            "category": pl.Series(["a", "b", "a", "b", "a", "b"], dtype=dtype),
        }
    )
    y = pl.Series("target", [0, 0, 0, 0, 1, 1])
    sampler = SMOTENC(
        categorical_features=categorical_features, k_neighbors=1, random_state=0
    )
    if categorical_dtype == "string" and categorical_features == "auto":
        with pytest.raises(ValueError, match="requires some categorical features"):
            sampler.fit_resample(X, y)
        return
    X_expected, y_expected = SMOTENC(
        categorical_features=[1],
        k_neighbors=1,
        random_state=0,
    ).fit_resample(X.to_numpy(), y.to_numpy())
    X_res, y_res = sampler.fit_resample(X, y)
    assert X_res.schema == X.schema
    assert_allclose(X_res["value"].to_numpy(), X_expected[:, 0].astype(float))
    assert_array_equal(X_res["category"].to_numpy(), X_expected[:, 1])
    assert_array_equal(y_res.to_numpy(), y_expected)
    assert_array_equal(sampler.categorical_features_, [1])


def test_smoten_polars(pl):
    X = pl.DataFrame({"category": ["a", "b", "a", "b", "a", "a"]})
    y = pl.Series("target", ["majority"] * 4 + ["minority"] * 2, dtype=pl.Categorical)
    X_res, y_res = SMOTEN(k_neighbors=1, random_state=0).fit_resample(X, y)
    assert X_res.schema == X.schema
    assert y_res.dtype == y.dtype
    assert y_res.name == y.name
    assert y_res.to_list().count("minority") == 4


def test_sampler_polars_binarized_target(pl):
    X = pl.DataFrame({"value": np.arange(9, dtype=float)})
    y = pl.DataFrame(
        np.eye(3, dtype=np.int8)[[0, 0, 0, 0, 1, 1, 1, 2, 2]],
        schema=["first", "second", "third"],
        orient="row",
    )
    sampler = RandomOverSampler(random_state=0)
    X_res, y_res = sampler.fit_resample(X, y)
    assert y_res.schema == y.schema
    assert_array_equal(y_res.to_numpy().sum(axis=0), [4, 4, 4])
    assert_array_equal(y_res.to_numpy().sum(axis=1), np.ones(12))
    assert X_res.equals(X[sampler.sample_indices_])


def test_arrays_transformer_polars_sparse(pl):
    from scipy.sparse import csr_matrix

    X = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int16), "b": [0.1, 0.2]})
    y = pl.Series("label", [0, 1], dtype=pl.Int8)
    X_res, y_res = ArraysTransformer(X, y).transform(
        csr_matrix(X.to_numpy()), y.to_numpy()
    )
    assert X_res.equals(X)
    assert X_res.schema == X.schema
    assert y_res.equals(y)


def test_polars_numeric_dtypes(pl):
    X = pl.DataFrame(
        {
            "integer": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Int16),
            "float": pl.Series([0.1, 0.3, 0.2, 0.5, 0.6, 0.7], dtype=pl.Float32),
        }
    )
    y = pl.Series("target", [0, 0, 0, 0, 1, 1], dtype=pl.UInt8)
    X_res, y_res = SMOTE(k_neighbors=1, random_state=0).fit_resample(X, y)
    assert X_res.schema == X.schema
    assert y_res.dtype == y.dtype
    X_expected, y_expected = SMOTE(k_neighbors=1, random_state=0).fit_resample(
        X.to_numpy(), y.to_numpy()
    )
    assert_array_equal(X_res["integer"].to_numpy(), X_expected[:, 0].astype(np.int16))
    assert_allclose(X_res["float"].to_numpy(), X_expected[:, 1].astype(np.float32))
    assert_array_equal(y_res.to_numpy(), y_expected)


@pytest.mark.parametrize("values", [[1, 2.5], [2.5, 1]])
def test_function_sampler_polars_object_numeric(pl, values):
    X = pl.DataFrame({"value": [1.0, 2.0]})
    y = pl.Series("target", [0, 1])
    sampler = FunctionSampler(
        func=lambda X, y: (np.array(values, dtype=object).reshape(-1, 1), y)
    )
    X_res, y_res = sampler.fit_resample(X, y)
    assert X_res.schema == X.schema
    assert_allclose(X_res["value"].to_numpy(), values)
    assert y_res.equals(y)


def test_polars_pipeline(pl, classification_data):
    X, y = classification_data
    pipeline = make_pipeline(
        StandardScaler().set_output(transform="polars"),
        SMOTE(random_state=0),
        LogisticRegression(),
    )
    pipeline.fit(X, y)
    classifier = pipeline[-1]
    assert_array_equal(classifier.feature_names_in_, X.columns)
    assert pipeline.predict(X).shape == y.shape
    X_res, y_res = pipeline[:-1].fit_resample(X, y)
    assert isinstance(X_res, pl.DataFrame)
    assert isinstance(y_res, pl.Series)
    assert X_res.schema == X.schema


def test_polars_output_without_pandas(pl, classification_data, monkeypatch):
    import builtins
    import sys

    X, y = classification_data
    original_import = builtins.__import__

    def import_without_pandas(name, *args, **kwargs):
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError("pandas is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pandas", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_pandas)
    X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
    assert isinstance(X_res, pl.DataFrame)
    assert isinstance(y_res, pl.Series)
