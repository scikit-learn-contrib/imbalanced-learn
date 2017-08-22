.. _developers-utils:

========================
Utilities for Developers
========================

Imbalanced-learn contains a number of utilities to help with development. These are
located in :mod:`imblearn.utils`, and include tools in a number of categories.
All the following functions and classes are in the module :mod:`imblearn.utils`.

.. warning ::

   These utilities are meant to be used internally within the imbalanced-learn
   package. They are not guaranteed to be stable between versions of
   imbalance-learn. Backports, in particular, will be removed as the
   imbalance-learn dependencies evolve.

.. currentmodule:: imblearn.utils

Validation Tools
================

These are tools used to check and validate input. When you write a function
which accepts arrays, matrices, or sparse matrices as arguments, the following
should be used when applicable.

- :func:`check_neighbors_object`: Check the objects is consistent to be a NN.
- :func:`check_target_type`: Check the target types to be conform to the current samplers.
- :func:`check_ratio`: Checks ratio for consistent type and return a dictionary
    containing each targeted class with its corresponding number of pixel.

Deprecation
-----------

.. warning ::
   Apart from :func:`deprecate_parameter` the rest of this section is taken from
   scikit-learn. Please refer to their original documentation.

If any publicly accessible method, function, attribute or parameter
is renamed, we still support the old one for two releases and issue
a deprecation warning when it is called/passed/accessed.
E.g., if the function ``zero_one`` is renamed to ``zero_one_loss``,
we add the decorator ``deprecated`` (from ``sklearn.utils``)
to ``zero_one`` and call ``zero_one_loss`` from that function::

    from ..utils import deprecated

    def zero_one_loss(y_true, y_pred, normalize=True):
        # actual implementation
        pass

    @deprecated("Function 'zero_one' was renamed to 'zero_one_loss' "
                "in version 0.13 and will be removed in release 0.15. "
                "Default behavior is changed from 'normalize=False' to "
                "'normalize=True'")
    def zero_one(y_true, y_pred, normalize=False):
        return zero_one_loss(y_true, y_pred, normalize)

If an attribute is to be deprecated,
use the decorator ``deprecated`` on a property.
E.g., renaming an attribute ``labels_`` to ``classes_`` can be done as::

    @property
    @deprecated("Attribute labels_ was deprecated in version 0.13 and "
                "will be removed in 0.15. Use 'classes_' instead")
    def labels_(self):
        return self.classes_

If a parameter has to be deprecated, use ``DeprecationWarning`` appropriately.
In the following example, k is deprecated and renamed to n_clusters::

    import warnings

    def example_function(n_clusters=8, k=None):
        if k is not None:
            warnings.warn("'k' was renamed to n_clusters in version 0.13 and "
                          "will be removed in 0.15.", DeprecationWarning)
            n_clusters = k

As in these examples, the warning message should always give both the
version in which the deprecation happened and the version in which the
old behavior will be removed. If the deprecation happened in version
0.x-dev, the message should say deprecation occurred in version 0.x and
the removal will be in 0.(x+2). For example, if the deprecation happened
in version 0.18-dev, the message should say it happened in version 0.18
and the old behavior will be removed in version 0.20.

In addition, a deprecation note should be added in the docstring, recalling the
same information as the deprecation warning as explained above. Use the
``.. deprecated::`` directive::

  .. deprecated:: 0.13
     ``k`` was renamed to ``n_clusters`` in version 0.13 and will be removed
     in 0.15.

On the top of all the functionality provided by scikit-learn. Imbalance-learn
provides :func:`deprecate_parameter`: which is used to deprecate a sampler's
parameter (attribute) by another one.
