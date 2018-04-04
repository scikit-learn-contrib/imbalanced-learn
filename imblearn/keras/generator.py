"""Implement generators for ``keras`` which will balance the data."""

import keras

from sklearn.base import clone
from sklearn.utils import safe_indexing

from ..under_sampling import RandomUnderSampler


class BalancedBatchGenerator(keras.utils.Sequence):
    """

    """
    def __init__(self, X, y, sampler=None, batch_size=64, stratify=True):
        self.X = X
        self.y = y
        self.sampler = sampler
        self.batch_size = batch_size
        self.stratify = stratify
        self._sample()

    def _sample(self):
        if self.sampler is None:
            self.sampler_ = RandomUnderSampler(return_indices=True)
        else:
            if not hasattr(self.sampler, 'return_indices'):
                raise ValueError("'sampler' needs to return the indices of "
                                 "the samples selected. Provide a sampler "
                                 "which has an attribute 'return_indices'.")
            self.sampler_ = clone(self.sampler)
            self.sampler_.set_params(return_indices=True)

        _, _, self.indices_ = self.sampler_.fit_sample(self.X, self.y)

    def __len__(self):
        return int(self.indices_.size // self.batch_size)

    def __getitem__(self, index):
        return (safe_indexing(self.X,
                              self.indices_[index * self.batch_size:
                                            (index + 1) * self.batch_size]),
                safe_indexing(self.y,
                              self.indices_[index * self.batch_size:
                                            (index + 1) * self.batch_size]))


# def balanced_batch_generator(X, y, sampler=None, batch_size=64,
#                              stratify=True):
#     """Create a balanced batch generator which can be plugged in
#     ``keras.fit_genertor``.

#     Parameters
#     ----------

#     """
#     if sampler is None:
#         sampler = RandomUnderSampler()
#     else:
#         if not hasattr(sampler, 'return_indices'):
#             raise ValueError("'sampler' needs to return the indices of "
#                              "the samples selected. Provide a sampler which "
#                              "has an attribute 'return_indices'.")
#         sampler.set_params(return_indices=True)

#     def generator(X=X, y=y, indices=indices, batch_size=batch_size,
#                   stratify=stratify):


#     _, _, indices = sampler.fit_sample(X, y)
