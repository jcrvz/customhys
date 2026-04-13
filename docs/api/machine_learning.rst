Machine Learning
================

.. automodule:: customhys.machine_learning
   :members:
   :undoc-members:
   :show-inheritance:

   This module provides Machine Learning utilities that can power a neural-network-based
   hyper-heuristic. It wraps TensorFlow models and includes data-processing pipelines
   for operator-sequence prediction.

   .. note::

      This module requires TensorFlow. Install it with ``pip install customhys[ml]``.

   **Key classes:**

   - ``DatasetSequences`` — converts raw operator sequences and fitness values into
     training samples suitable for supervised learning.
   - ``ModelPredictor`` — a configurable feed-forward neural network for predicting the
     next operator in a sequence.

   **Quick usage:**

   .. code-block:: python

      from customhys.machine_learning import DatasetSequences, ModelPredictor

      ds = DatasetSequences(sequences, fitnesses, num_operators=13)
      model = ModelPredictor(num_operators=13)
      model.train(ds)
