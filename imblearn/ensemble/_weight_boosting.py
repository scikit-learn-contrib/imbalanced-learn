import numbers
from copy import deepcopy

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.base import _set_random_states
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import safe_indexing

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline


class RUSBoostClassifier(AdaBoostClassifier):
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.,
                 algorithm='SAMME.R', sampling_strategy='auto',
                 replacement=False, random_state=None):
        super(RUSBoostClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state)
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement

    def fit(self, X, y, sample_weight=None):
        self.samplers_ = []
        self.pipelines_ = []
        super(RUSBoostClassifier, self).fit(X, y, sample_weight)
        return self

    def _validate_estimator(self, default=DecisionTreeClassifier()):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            self.base_estimator_ = clone(self.base_estimator)
        else:
            self.base_estimator_ = clone(default)

        self.base_sampler_ = RandomUnderSampler(
            sampling_strategy=self.sampling_strategy,
            replacement=self.replacement,
            return_indices=True)

    def _make_sampler_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(**dict((p, getattr(self, p))
                                    for p in self.estimator_params))
        sampler = clone(self.base_sampler_)

        if random_state is not None:
            _set_random_states(estimator, random_state)
            _set_random_states(sampler, random_state)

        if append:
            self.estimators_.append(estimator)
            self.samplers_.append(sampler)
            self.pipelines_.append(make_pipeline(deepcopy(sampler),
                                                 deepcopy(estimator)))
            # do not return the indices within a pipeline
            self.pipelines_[-1].named_steps['randomundersampler'].set_params(
                return_indices=False)

        return estimator, sampler

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator, sampler = self._make_sampler_estimator(
            random_state=random_state)

        X_res, y_res, idx_res = sampler.fit_resample(X, y)
        if sample_weight is not None:
            sample_weight_res = safe_indexing(sample_weight, idx_res)
        else:
            sample_weight_res = None
        estimator.fit(X_res, y_res, sample_weight=sample_weight_res)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * self.learning_rate
                            * ((n_classes - 1.) / n_classes)
                            * (y_coding * np.log(y_predict_proba)).sum(axis=1))

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, 1., estimator_error

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator, sampler = self._make_sampler_estimator(
            random_state=random_state)

        X_res, y_res, idx_res = sampler.fit_resample(X, y)
        if sample_weight is not None:
            sample_weight_res = safe_indexing(sample_weight, idx_res)
        else:
            sample_weight_res = None
        estimator.fit(X_res, y_res, sample_weight=sample_weight_res)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            self.samplers_.pop(-1)
            self.pipelines_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, estimator_weight, estimator_error
