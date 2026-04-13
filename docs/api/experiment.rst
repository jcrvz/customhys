Experiment
==========

.. automodule:: customhys.experiment
   :members:
   :undoc-members:
   :show-inheritance:

   The ``Experiment`` class orchestrates batch hyper-heuristic runs across multiple
   benchmark functions and dimensionalities.

   Configuration can be provided via a JSON file in ``exconf/`` or as Python
   dictionaries.

   **Quick usage:**

   .. code-block:: python

      from customhys.experiment import Experiment

      # From a JSON configuration file
      exp = Experiment(config_file="demo.json")
      exp.run()

      # Or programmatically
      exp = Experiment(
          exp_config={
              "experiment_name": "quick",
              "experiment_type": "default",
              "heuristic_collection_file": "default.txt",
          },
          hh_config={"cardinality": 3, "num_replicas": 30},
          prob_config={"dimensions": [2, 5], "functions": ["Sphere"]},
      )
      exp.run()
