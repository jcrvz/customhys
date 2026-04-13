Search Operators
================

.. automodule:: customhys.operators
   :members:
   :undoc-members:
   :show-inheritance:

   This module provides a library of **13 search operators** extracted from well-known
   metaheuristics. Each operator is a function that modifies a ``Population`` object's
   positions *in place*.

   Operators are specified as 3-tuples for use in the Metaheuristic and Hyperheuristic
   classes::

       (operator_name, parameters_dict, selector_name)

   **Example:**

   .. code-block:: python

      from customhys.population import Population
      from customhys import operators as op

      pop = Population(boundaries=([-5]*10, [5]*10), num_agents=30)
      pop.initialise_positions("random")

      # Apply a random-search perturbation
      op.random_search(pop, scale=0.01, distribution="uniform")
