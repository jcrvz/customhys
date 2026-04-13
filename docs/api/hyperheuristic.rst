Hyper-heuristic
===============

.. automodule:: customhys.hyperheuristic
   :members:
   :undoc-members:
   :show-inheritance:

   The ``Hyperheuristic`` class explores the space of possible metaheuristics to find
   the best operator sequence for a given problem. The search is guided by a Simulated
   Annealing acceptance criterion.

   When TensorFlow is available, the hyper-heuristic can also leverage a neural-network
   predictor (see :mod:`customhys.machine_learning`) to bias the operator selection.

   **Quick usage:**

   .. code-block:: python

      from customhys import benchmark_func as bf
      from customhys.hyperheuristic import Hyperheuristic

      func = bf.Rastrigin(10)
      prob = {
          "function": func.get_func_val,
          "is_constrained": True,
          "boundaries": func.get_search_range(),
      }

      hh = Hyperheuristic(
          heuristic_space="default.txt",
          problem=prob,
          parameters={
              "cardinality": 3,
              "num_iterations": 100,
              "num_agents": 30,
              "num_replicas": 30,
              "num_steps": 100,
              "stagnation_percentage": 0.3,
              "max_temperature": 200,
              "cooling_rate": 0.05,
          },
      )
      hh.run()
