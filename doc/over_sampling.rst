.. _over-sampling:

=============
Over-sampling
=============

.. currentmodule:: imblearn.over_sampling

.. image:: ./modules/balancing_problem/linear_svc_imbalanced_issue.png
   :scale: 80
   :align: center

One way to fight the problem of imbalanced data set is to generate new samples
in the classes which are under-represented. The most naive strategy is to
balance those classes by randomly sampling with replacement.
