Metaheuristic
=============

.. automodule:: customhys.metaheuristic
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Population, Operators

   A ``Metaheuristic`` object combines a problem, a population, and a sequence of
   search operators into an iterative optimisation procedure.

   .. note::

      ``Population`` and ``Operators`` are re-exported by this module for convenience
      but are fully documented in their own pages:
      :doc:`population` and :doc:`operators`.

   **Lifecycle:**

   1. Initialise the population (``apply_initialiser``).
   2. At each iteration, apply every operator in sequence (``apply_search_operator``).
   3. After the run, retrieve the best solution with ``get_solution()``.

   **Quick usage:**

   .. code-block:: python

      from customhys import benchmark_func as bf
      from customhys.metaheuristic import Metaheuristic

      func = bf.Sphere(5)
      prob = {
          "function": func.get_func_val,
          "is_constrained": True,
          "boundaries": func.get_search_range(),
      }
      ops = [
          ("random_search", {"scale": 0.01, "distribution": "uniform"}, "greedy"),
      ]
      mh = Metaheuristic(prob, ops, num_agents=30, num_iterations=100)
      mh.run()
      pos, fit = mh.get_solution()
