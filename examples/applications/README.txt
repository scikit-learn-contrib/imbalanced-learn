Resampling comparison
---------------------

The plot_resampling_comparison.py example uses an imbalanced version
of the Breast Cancer Wisconsin dataset to compare four approaches:

- No resampling
- Random oversampling
- Random undersampling
- SMOTE

Each approach uses standardization and logistic regression. Resampling
is applied only to the training data through an imbalanced-learn pipeline.

The example reports balanced accuracy, minority-class F1 for malignant
cases (class label 0), and geometric mean.

To run the example from the repository root::

    python examples/applications/plot_resampling_comparison.py

The example prints a results table and displays a bar chart. The chart's
y-axis starts at 0.90 to highlight small differences, so it should be
interpreted together with the table.

Results are reproducible using a fixed random state. This example shows
that resampling does not necessarily improve every metric. Results from
this small test set should not be generalized to other datasets.

.. _realword_examples:

Examples based on real world datasets
-------------------------------------

Examples that use real-world datasets.
