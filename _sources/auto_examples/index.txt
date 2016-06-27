.. _general_examples:

General examples
----------------

General-purpose and introductory examples for the `imbalanced-learn` toolbox.
.. raw:: html

    <div style='clear:both'></div>

.. _combine_examples:

Examples using combine class methods
====================================

Combine methods mixed over- and under-sampling methods. Generally SMOTE is used for over-sampling while some cleaning methods (i.e., ENN and Tomek links) are used to under-sample.

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the SMOTE + ENN method.">

.. only:: html

    .. figure:: /auto_examples/combine/images/thumb/sphx_glr_plot_smote_enn_thumb.png

        :ref:`sphx_glr_auto_examples_combine_plot_smote_enn.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/combine/plot_smote_enn

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the SMOTE + Tomek method.">

.. only:: html

    .. figure:: /auto_examples/combine/images/thumb/sphx_glr_plot_smote_tomek_thumb.png

        :ref:`sphx_glr_auto_examples_combine_plot_smote_tomek.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/combine/plot_smote_tomek
.. raw:: html

    <div style='clear:both'></div>

.. _ensemble_examples:

Example using ensemble class methods
====================================

Under-sampling methods implies that samples of the majority class are lost during the balancing procedure.
Ensemble methods offer an alternative to use most of the samples.
In fact, an ensemble of balanced sets is created and used to later train any classifier.

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the easy ensemble method.">

.. only:: html

    .. figure:: /auto_examples/ensemble/images/thumb/sphx_glr_plot_easy_ensemble_thumb.png

        :ref:`sphx_glr_auto_examples_ensemble_plot_easy_ensemble.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/ensemble/plot_easy_ensemble

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the balance cascade ensemble method.">

.. only:: html

    .. figure:: /auto_examples/ensemble/images/thumb/sphx_glr_plot_balance_cascade_thumb.png

        :ref:`sphx_glr_auto_examples_ensemble_plot_balance_cascade.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/ensemble/plot_balance_cascade
.. raw:: html

    <div style='clear:both'></div>

.. _over_sampling_examples:

Example using over-sampling class methods
=========================================

Data balancing can be performed by over-sampling such that new samples are generated in the minority class to reach a given balancing ratio.

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the random SMOTE SVM method.">

.. only:: html

    .. figure:: /auto_examples/over-sampling/images/thumb/sphx_glr_plot_smote_svm_thumb.png

        :ref:`sphx_glr_auto_examples_over-sampling_plot_smote_svm.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/over-sampling/plot_smote_svm

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the Adaptive Synthetic Sampling Approach for Imbalanced Learning ADASYN meth...">

.. only:: html

    .. figure:: /auto_examples/over-sampling/images/thumb/sphx_glr_plot_adasyn_thumb.png

        :ref:`sphx_glr_auto_examples_over-sampling_plot_adasyn.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/over-sampling/plot_adasyn

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the random SMOTE regular method.">

.. only:: html

    .. figure:: /auto_examples/over-sampling/images/thumb/sphx_glr_plot_smote_thumb.png

        :ref:`sphx_glr_auto_examples_over-sampling_plot_smote.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/over-sampling/plot_smote

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the random SMOTE borderline 1 method.">

.. only:: html

    .. figure:: /auto_examples/over-sampling/images/thumb/sphx_glr_plot_smote_bordeline_1_thumb.png

        :ref:`sphx_glr_auto_examples_over-sampling_plot_smote_bordeline_1.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/over-sampling/plot_smote_bordeline_1

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the random SMOTE borderline 2 method.">

.. only:: html

    .. figure:: /auto_examples/over-sampling/images/thumb/sphx_glr_plot_smote_bordeline_2_thumb.png

        :ref:`sphx_glr_auto_examples_over-sampling_plot_smote_bordeline_2.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/over-sampling/plot_smote_bordeline_2

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the random over-sampling method.">

.. only:: html

    .. figure:: /auto_examples/over-sampling/images/thumb/sphx_glr_plot_random_over_sampling_thumb.png

        :ref:`sphx_glr_auto_examples_over-sampling_plot_random_over_sampling.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/over-sampling/plot_random_over_sampling
.. raw:: html

    <div style='clear:both'></div>

.. _pipeline_examples:

Pipeline examples
=================

Example of how to use the a pipeline to include under-sampling with `scikit-learn` estimators.
.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example of the Pipeline object working with transformers and resamplers.">

.. only:: html

    .. figure:: /auto_examples/pipeline/images/thumb/sphx_glr_plot_pipeline_classification_thumb.png

        :ref:`sphx_glr_auto_examples_pipeline_plot_pipeline_classification.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/pipeline/plot_pipeline_classification
.. raw:: html

    <div style='clear:both'></div>

.. _under_sampling_examples:

Example using under-sampling class methods
==========================================

Under-sampling refers to the process of reducing the number of samples in the majority classes.
The implemented methods can be categorized into 2 groups: (i) fixed under-sampling and (ii) cleaning under-sampling.

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the nearmiss 1 method.">

.. only:: html

    .. figure:: /auto_examples/under-sampling/images/thumb/sphx_glr_plot_nearmiss_1_thumb.png

        :ref:`sphx_glr_auto_examples_under-sampling_plot_nearmiss_1.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/under-sampling/plot_nearmiss_1

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the nearmiss 1 method.">

.. only:: html

    .. figure:: /auto_examples/under-sampling/images/thumb/sphx_glr_plot_nearmiss_2_thumb.png

        :ref:`sphx_glr_auto_examples_under-sampling_plot_nearmiss_2.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/under-sampling/plot_nearmiss_2

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the nearmiss 1 method.">

.. only:: html

    .. figure:: /auto_examples/under-sampling/images/thumb/sphx_glr_plot_nearmiss_3_thumb.png

        :ref:`sphx_glr_auto_examples_under-sampling_plot_nearmiss_3.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/under-sampling/plot_nearmiss_3

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the Tomek links method.">

.. only:: html

    .. figure:: /auto_examples/under-sampling/images/thumb/sphx_glr_plot_tomek_links_thumb.png

        :ref:`sphx_glr_auto_examples_under-sampling_plot_tomek_links.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/under-sampling/plot_tomek_links

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the cluster centroids method.">

.. only:: html

    .. figure:: /auto_examples/under-sampling/images/thumb/sphx_glr_plot_cluster_centroids_thumb.png

        :ref:`sphx_glr_auto_examples_under-sampling_plot_cluster_centroids.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/under-sampling/plot_cluster_centroids

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the one-sided selection method.">

.. only:: html

    .. figure:: /auto_examples/under-sampling/images/thumb/sphx_glr_plot_one_sided_selection_thumb.png

        :ref:`sphx_glr_auto_examples_under-sampling_plot_one_sided_selection.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/under-sampling/plot_one_sided_selection

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the random under-sampling method.">

.. only:: html

    .. figure:: /auto_examples/under-sampling/images/thumb/sphx_glr_plot_random_under_sampler_thumb.png

        :ref:`sphx_glr_auto_examples_under-sampling_plot_random_under_sampler.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/under-sampling/plot_random_under_sampler

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the edited nearest-neighbours method.">

.. only:: html

    .. figure:: /auto_examples/under-sampling/images/thumb/sphx_glr_plot_edited_nearest_neighbours_thumb.png

        :ref:`sphx_glr_auto_examples_under-sampling_plot_edited_nearest_neighbours.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/under-sampling/plot_edited_nearest_neighbours

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the neighbourhood cleaning rule method.">

.. only:: html

    .. figure:: /auto_examples/under-sampling/images/thumb/sphx_glr_plot_neighbourhood_cleaning_rule_thumb.png

        :ref:`sphx_glr_auto_examples_under-sampling_plot_neighbourhood_cleaning_rule.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/under-sampling/plot_neighbourhood_cleaning_rule

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the condensed nearest-neighbour method.">

.. only:: html

    .. figure:: /auto_examples/under-sampling/images/thumb/sphx_glr_plot_condensed_nearest_neighbour_thumb.png

        :ref:`sphx_glr_auto_examples_under-sampling_plot_condensed_nearest_neighbour.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/under-sampling/plot_condensed_nearest_neighbour

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the instance hardness threshold method.">

.. only:: html

    .. figure:: /auto_examples/under-sampling/images/thumb/sphx_glr_plot_instance_hardness_threshold_thumb.png

        :ref:`sphx_glr_auto_examples_under-sampling_plot_instance_hardness_threshold.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/under-sampling/plot_instance_hardness_threshold

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An illustration of the repeated edited nearest-neighbours method.">

.. only:: html

    .. figure:: /auto_examples/under-sampling/images/thumb/sphx_glr_plot_repeated_edited_nearest_neighbours_thumb.png

        :ref:`sphx_glr_auto_examples_under-sampling_plot_repeated_edited_nearest_neighbours.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/under-sampling/plot_repeated_edited_nearest_neighbours
.. raw:: html

    <div style='clear:both'></div>

