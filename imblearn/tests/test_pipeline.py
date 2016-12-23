"""
Test the pipeline module.
"""
import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.testing import (
    assert_array_almost_equal, assert_array_equal, assert_equal, assert_false,
    assert_raise_message, assert_raises, assert_raises_regex, assert_true,
    assert_warns_message)

from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import (RandomUnderSampler,
                                     EditedNearestNeighbours as ENN)

JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger", )


class IncorrectT(object):
    """Small class to test parameter dispatching.
    """

    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class T(IncorrectT):
    def fit(self, X, y):
        return self

    def get_params(self, deep=False):
        return {'a': self.a, 'b': self.b}

    def set_params(self, **params):
        self.a = params['a']
        return self


class TransfT(T):
    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X


class FitParamT(object):
    """Mock classifier
    """

    def __init__(self):
        self.successful = False

    def fit(self, X, y, should_succeed=False):
        self.successful = should_succeed

    def predict(self, X):
        return self.successful


class FitTransformSample(T):
    """Mock classifier
    """

    def fit(self, X, y, should_succeed=False):
        pass

    def sample(self, X, y=None):
        return X, y

    def transform(self, X, y=None):
        return X


def test_pipeline_init():
    # Test the various init parameters of the pipeline.
    assert_raises(TypeError, Pipeline)
    # Check that we can't instantiate pipelines with objects without fit
    # method
    pipe = assert_raises(TypeError, Pipeline, [('svc', IncorrectT)])
    # Smoke test with only an estimator
    clf = T()
    pipe = Pipeline([('svc', clf)])
    assert_equal(
        pipe.get_params(deep=True),
        dict(
            svc__a=None, svc__b=None, svc=clf, **pipe.get_params(deep=False)))

    # Check that params are set
    pipe.set_params(svc__a=0.1)
    assert_equal(clf.a, 0.1)
    assert_equal(clf.b, None)
    # Smoke test the repr:
    repr(pipe)

    # Test with two objects
    clf = SVC()
    filter1 = SelectKBest(f_classif)
    pipe = Pipeline([('anova', filter1), ('svc', clf)])

    # Check that we can't use the same stage name twice
    assert_raises(ValueError, Pipeline, [('svc', SVC()), ('svc', SVC())])

    # Check that params are set
    pipe.set_params(svc__C=0.1)
    assert_equal(clf.C, 0.1)
    # Smoke test the repr:
    repr(pipe)

    # Check that params are not set when naming them wrong
    assert_raises(ValueError, pipe.set_params, anova__C=0.1)

    # Test clone
    pipe2 = clone(pipe)
    assert_false(pipe.named_steps['svc'] is pipe2.named_steps['svc'])

    # Check that apart from estimators, the parameters are the same
    params = pipe.get_params(deep=True)
    params2 = pipe2.get_params(deep=True)

    for x in pipe.get_params(deep=False):
        params.pop(x)

    for x in pipe2.get_params(deep=False):
        params2.pop(x)

    # Remove estimators that where copied
    params.pop('svc')
    params.pop('anova')
    params2.pop('svc')
    params2.pop('anova')
    assert_equal(params, params2)


def test_pipeline_methods_anova():
    # Test the various methods of the pipeline (anova).
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Test with Anova + LogisticRegression
    clf = LogisticRegression()
    filter1 = SelectKBest(f_classif, k=2)
    pipe = Pipeline([('anova', filter1), ('logistic', clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_fit_params():
    # Test that the pipeline can take fit parameters
    pipe = Pipeline([('transf', TransfT()), ('clf', FitParamT())])
    pipe.fit(X=None, y=None, clf__should_succeed=True)
    # classifier should return True
    assert_true(pipe.predict(None))
    # and transformer params should not be changed
    assert_true(pipe.named_steps['transf'].a is None)
    assert_true(pipe.named_steps['transf'].b is None)


def test_pipeline_raise_set_params_error():
    # Test pipeline raises set params error message for nested models.
    pipe = Pipeline([('cls', LinearRegression())])

    # expected error message
    error_msg = ('Invalid parameter %s for estimator %s. '
                 'Check the list of available parameters '
                 'with `estimator.get_params().keys()`.')

    assert_raise_message(
        ValueError,
        error_msg % ('fake', 'Pipeline'),
        pipe.set_params,
        fake='nope')

    # nested model check
    assert_raise_message(
        ValueError,
        error_msg % ("fake", pipe),
        pipe.set_params,
        fake__estimator='nope')


def test_pipeline_methods_pca_svm():
    # Test the various methods of the pipeline (pca + svm).
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Test with PCA + SVC
    clf = SVC(probability=True, random_state=0)
    pca = PCA()
    pipe = Pipeline([('pca', pca), ('svc', clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_methods_preprocessing_svm():
    # Test the various methods of the pipeline (preprocessing + svm).
    iris = load_iris()
    X = iris.data
    y = iris.target
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    clf = SVC(probability=True, random_state=0, decision_function_shape='ovr')

    for preprocessing in [scaler, pca]:
        pipe = Pipeline([('preprocess', preprocessing), ('svc', clf)])
        pipe.fit(X, y)

        # check shapes of various prediction functions
        predict = pipe.predict(X)
        assert_equal(predict.shape, (n_samples, ))

        proba = pipe.predict_proba(X)
        assert_equal(proba.shape, (n_samples, n_classes))

        log_proba = pipe.predict_log_proba(X)
        assert_equal(log_proba.shape, (n_samples, n_classes))

        decision_function = pipe.decision_function(X)
        assert_equal(decision_function.shape, (n_samples, n_classes))

        pipe.score(X, y)


def test_fit_predict_on_pipeline():
    # test that the fit_predict method is implemented on a pipeline
    # test that the fit_predict on pipeline yields same results as applying
    # transform and clustering steps separately
    iris = load_iris()
    scaler = StandardScaler()
    km = KMeans(random_state=0)

    # first compute the transform and clustering step separately
    scaled = scaler.fit_transform(iris.data)
    separate_pred = km.fit_predict(scaled)

    # use a pipeline to do the transform and clustering in one step
    pipe = Pipeline([('scaler', scaler), ('Kmeans', km)])
    pipeline_pred = pipe.fit_predict(iris.data)

    assert_array_almost_equal(pipeline_pred, separate_pred)


def test_fit_predict_on_pipeline_without_fit_predict():
    # tests that a pipeline does not have fit_predict method when final
    # step of pipeline does not have fit_predict defined
    scaler = StandardScaler()
    pca = PCA()
    pipe = Pipeline([('scaler', scaler), ('pca', pca)])
    assert_raises_regex(AttributeError,
                        "'PCA' object has no attribute 'fit_predict'", getattr,
                        pipe, 'fit_predict')


def test_pipeline_transform():
    # Test whether pipeline works with a transformer at the end.
    # Also test pipeline.transform and pipeline.inverse_transform
    iris = load_iris()
    X = iris.data
    pca = PCA(n_components=2)
    pipeline = Pipeline([('pca', pca)])

    # test transform and fit_transform:
    X_trans = pipeline.fit(X).transform(X)
    X_trans2 = pipeline.fit_transform(X)
    X_trans3 = pca.fit_transform(X)
    assert_array_almost_equal(X_trans, X_trans2)
    assert_array_almost_equal(X_trans, X_trans3)

    X_back = pipeline.inverse_transform(X_trans)
    X_back2 = pca.inverse_transform(X_trans)
    assert_array_almost_equal(X_back, X_back2)


def test_pipeline_fit_transform():
    # Test whether pipeline works with a transformer missing fit_transform
    iris = load_iris()
    X = iris.data
    y = iris.target
    transft = TransfT()
    pipeline = Pipeline([('mock', transft)])

    # test fit_transform:
    X_trans = pipeline.fit_transform(X, y)
    X_trans2 = transft.fit(X, y).transform(X)
    assert_array_almost_equal(X_trans, X_trans2)


def test_make_pipeline():
    t1 = TransfT()
    t2 = TransfT()

    pipe = make_pipeline(t1, t2)
    assert_true(isinstance(pipe, Pipeline))
    assert_equal(pipe.steps[0][0], "transft-1")
    assert_equal(pipe.steps[1][0], "transft-2")

    pipe = make_pipeline(t1, t2, FitParamT())
    assert_true(isinstance(pipe, Pipeline))
    assert_equal(pipe.steps[0][0], "transft-1")
    assert_equal(pipe.steps[1][0], "transft-2")
    assert_equal(pipe.steps[2][0], "fitparamt")


def test_classes_property():
    iris = load_iris()
    X = iris.data
    y = iris.target

    reg = make_pipeline(SelectKBest(k=1), LinearRegression())
    reg.fit(X, y)
    assert_raises(AttributeError, getattr, reg, "classes_")

    clf = make_pipeline(SelectKBest(k=1), LogisticRegression(random_state=0))
    assert_raises(AttributeError, getattr, clf, "classes_")
    clf.fit(X, y)
    assert_array_equal(clf.classes_, np.unique(y))


def test_X1d_inverse_transform():
    transformer = TransfT()
    pipeline = make_pipeline(transformer)
    X = np.ones(10)
    msg = "1d X will not be reshaped in pipeline.inverse_transform"
    assert_warns_message(FutureWarning, msg, pipeline.inverse_transform, X)


def test_pipeline_methods_pca_rus_svm():
    # Test the various methods of the pipeline (pca + svm).
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0)

    # Test with PCA + SVC
    clf = SVC(probability=True, random_state=0)
    pca = PCA()
    rus = RandomUnderSampler(random_state=0)
    pipe = Pipeline([('pca', pca), ('rus', rus), ('svc', clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_methods_rus_pca_svm():
    # Test the various methods of the pipeline (pca + svm).
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0)

    # Test with PCA + SVC
    clf = SVC(probability=True, random_state=0)
    pca = PCA()
    rus = RandomUnderSampler(random_state=0)
    pipe = Pipeline([('rus', rus), ('pca', pca), ('svc', clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_sample():
    # Test whether pipeline works with a sampler at the end.
    # Also test pipeline.sampler
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0)

    rus = RandomUnderSampler(random_state=0)
    pipeline = Pipeline([('rus', rus)])

    # test transform and fit_transform:
    X_trans, y_trans = pipeline.fit(X, y).sample(X, y)
    X_trans2, y_trans2 = pipeline.fit_sample(X, y)
    X_trans3, y_trans3 = rus.fit_sample(X, y)
    assert_array_almost_equal(X_trans, X_trans2)
    assert_array_almost_equal(X_trans, X_trans3)
    assert_array_almost_equal(y_trans, y_trans2)
    assert_array_almost_equal(y_trans, y_trans3)

    pca = PCA()
    pipeline = Pipeline([('pca', pca), ('rus', rus)])

    X_trans, y_trans = pipeline.fit(X, y).sample(X, y)
    X_pca = pca.fit_transform(X)
    X_trans2, y_trans2 = rus.fit_sample(X_pca, y)
    assert_array_almost_equal(X_trans, X_trans2)
    assert_array_almost_equal(y_trans, y_trans2)


def test_pipeline_sample_transform():
    # Test whether pipeline works with a sampler at the end.
    # Also test pipeline.sampler
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0)

    rus = RandomUnderSampler(random_state=0)
    pca = PCA()
    pca2 = PCA()
    pipeline = Pipeline([('pca', pca), ('rus', rus), ('pca2', pca2)])

    pipeline.fit(X, y).transform(X)


def test_pipeline_methods_anova_rus():
    # Test the various methods of the pipeline (anova).
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0)
    # Test with RandomUnderSampling + Anova + LogisticRegression
    clf = LogisticRegression()
    rus = RandomUnderSampler(random_state=0)
    filter1 = SelectKBest(f_classif, k=2)
    pipe = Pipeline([('rus', rus), ('anova', filter1), ('logistic', clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_with_step_that_implements_both_sample_and_transform():
    # Test the various methods of the pipeline (anova).
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0)

    clf = LogisticRegression()
    assert_raises(TypeError, Pipeline, [('step', FitTransformSample()),
                                        ('logistic', clf)])
    # assert_raises(TypeError, lambda x: [][0])


def test_pipeline_with_step_that_it_is_pipeline():
    # Test the various methods of the pipeline (anova).
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0)
    # Test with RandomUnderSampling + Anova + LogisticRegression
    clf = LogisticRegression()
    rus = RandomUnderSampler(random_state=0)
    filter1 = SelectKBest(f_classif, k=2)
    pipe1 = Pipeline([('rus', rus), ('anova', filter1)])
    assert_raises(TypeError, Pipeline, [('pipe1', pipe1), ('logistic', clf)])


def test_pipeline_fit_then_sample_with_sampler_last_estimator():
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=50000,
        random_state=0)

    rus = RandomUnderSampler(random_state=42)
    enn = ENN()
    pipeline = make_pipeline(rus, enn)
    X_fit_sample_resampled, y_fit_sample_resampled = pipeline.fit_sample(X, y)
    pipeline = make_pipeline(rus, enn)
    pipeline.fit(X, y)
    X_fit_then_sample_res, y_fit_then_sample_res = pipeline.sample(X, y)
    assert_array_equal(X_fit_sample_resampled, X_fit_then_sample_res)
    assert_array_equal(y_fit_sample_resampled, y_fit_then_sample_res)


def test_pipeline_fit_then_sample_3_samplers_with_sampler_last_estimator():
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=50000,
        random_state=0)

    rus = RandomUnderSampler(random_state=42)
    enn = ENN()
    pipeline = make_pipeline(rus, enn, rus)
    X_fit_sample_resampled, y_fit_sample_resampled = pipeline.fit_sample(X, y)
    pipeline = make_pipeline(rus, enn, rus)
    pipeline.fit(X, y)
    X_fit_then_sample_res, y_fit_then_sample_res = pipeline.sample(X, y)
    assert_array_equal(X_fit_sample_resampled, X_fit_then_sample_res)
    assert_array_equal(y_fit_sample_resampled, y_fit_then_sample_res)
