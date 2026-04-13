Population
==========

.. automodule:: customhys.population
   :members:
   :undoc-members:
   :show-inheritance:

   The ``Population`` class represents a collection of search agents within a bounded
   domain. It manages agent positions, velocities, fitness values, and selection
   strategies.

   **Initialisation schemes:** ``random``, ``sobol``, ``halton``, ``lhs``, ``vertex``,
   among others.

   **Selection methods:** ``greedy``, ``all``, ``metropolis``, ``probabilistic``.

   **Quick usage:**

   .. code-block:: python

      from customhys.population import Population

      pop = Population(boundaries=([-5]*10, [5]*10), num_agents=30)
      pop.initialise_positions("random")
