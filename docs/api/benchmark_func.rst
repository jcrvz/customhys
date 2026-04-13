Benchmark Functions
===================

.. automodule:: customhys.benchmark_func
   :members:
   :undoc-members:
   :show-inheritance:

   This module provides N-dimensional benchmark functions for continuous optimisation.
   Each function is implemented as a class inheriting from a common abstract base that
   exposes ``get_func_val``, ``get_search_range``, and ``get_optimal``.

   The catalogue covers classical test suites (Ackley, Rastrigin, Rosenbrock, Schwefel,
   Griewank, …) as well as the CEC 2005 competition suite when the ``optproblems``
   package is available.

   **Quick usage:**

   .. code-block:: python

      from customhys import benchmark_func as bf

      func = bf.Sphere(10)
      print(func.get_func_val([0]*10))  # 0.0
