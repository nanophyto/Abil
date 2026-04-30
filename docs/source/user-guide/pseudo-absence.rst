Pseudo-absence generation
*************************

In this example we use a small synthetic dataset to show pseudo-absence generation in environmental space.
Presences are restricted to warmer, lower-silicate conditions, and pseudo-absences are sampled outside the area of applicability.

Running the example
~~~~~~~~~~~~~~~~~~~

Before running the Python script we need to import the required packages and define the synthetic dataset.
The candidate rows span temperature from 0 to 25 and silicate from 0.1 to 3.
Observed presences are then selected from rows with temperature above 5 and silicate below 1.

.. literalinclude:: ../../../examples/pseudo_absence.py
   :lines: 1-32
   :language: python

Generating pseudo-absences
~~~~~~~~~~~~~~~~~~~~~~~~~~

Next we call ``generate_pseudo_absences``.
With ``absence_ratio=1``, the function targets one pseudo-absence for each observed presence.
If there are fewer candidate rows outside the area of applicability, rows are sampled with replacement by default.

.. literalinclude:: ../../../examples/pseudo_absence.py
   :lines: 34-41
   :language: python

Plotting
~~~~~~~~

Now that we have pseudo-absences we can plot them in environmental space:

.. literalinclude:: ../../../examples/pseudo_absence.py
   :lines: 43-73
   :language: python

.. figure:: ../../../examples/pseudo_absence.png
