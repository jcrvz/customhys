Tools
=====

.. automodule:: customhys.tools
   :members:
   :undoc-members:
   :show-inheritance:

   Utility functions used across the CUSTOMHyS package: JSON I/O, statistical
   summaries, data inspection helpers, and more.

   **Quick usage:**

   .. code-block:: python

      from customhys import tools as tl

      # Read a JSON results file
      data = tl.read_json("data_files/raw/results.json")

      # Inspect the data structure
      tl.printmsk(data)
