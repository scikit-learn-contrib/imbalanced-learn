.. _problem_statement:

=================
Problem statement
=================

The learning phase and the subsequent prediction of machine learning algorithms
can be affected by the problem of imbalanced data set. The balancing issue
corresponds to the difference of the number of samples in the different
classes. We illustrate the effect of training a linear SVM classifier with
different level of class balancing.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_001.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center

As expected, the decision function of the linear SVM is highly impacted. With a
greater imbalanced ratio, the decision function favor the class with the larger
number of samples, usually referred as the majority class.
