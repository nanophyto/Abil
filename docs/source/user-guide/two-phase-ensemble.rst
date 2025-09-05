Southern Ocean distribution of *Gephyrocapsa huxleyi*
*****************************************************

In this example we will use Abil to predict the biomass of a highly abundant calcifying nanoplankton (*Gephyrocapsa huxleyi*).
We will focus on the Southern Ocean, a region with high *G. huxleyi* stocks and of significant importance to the ocean carbon cycle.
This region has an interesting macroscale distribution of *G. huxleyi* biomass, with peak concentrations in the so called "great calcite belt" 
between ~40-60°S but a notable absence below ~60°S due to competition with silicifying nanoplankton :footcite:p:`nissen2018factors`.

For this example we will use a 2-phase regressor to better constrain absences below ~60°S.
For an example of the 1-phase model see :doc:`Coccolithophore abundance in the Bermuda Atlantic Time Series  <one-phase-ensemble>`. 

YAML example
~~~~~~~~~~~~

Before running the model, model specifications need to be defined in a YAML file. 
For a detailed explanation of each parameter see :ref:`yaml_config`.

An example of YAML file of a 2-phase model is provided below.
Note that compared to a 1-phase regressor model, the hyper-parameters for the classifier also need to be specified.

.. literalinclude:: ../../../examples/2-phase.yml
   :language: yaml

Running the model
~~~~~~~~~~~~~~~~~
After specifying the model configuration in the relevant YAML file, we can use the Abil API
to 1) tune the model, evaluating the model performance across different hyper-parameter values and then 
selecting the best configuration 2) predict in-sample and out-of-sample observations based on the optimal
hyper-parameter configuration identified in the first step 3) conduct post-processing such as exporting
relevant performance metrics, spatially or temporally integrated target estimates, and diversity metrics.


Loading dependencies
^^^^^^^^^^^^^^^^^^^^

Before running the Python script we need to import all relevant Python packages.
For instructions on how to install these packages, see `requirements.txt <../../../../../requirements.txt>`_
and the Abil :ref:`getting-started`.

.. literalinclude:: ../../../examples/2-phase.py
   :lines: 4-21
   :language: python

Loading the configuration YAML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After loading the required packages we need to define our file paths.

.. literalinclude:: ../../../examples/2-phase.py
   :lines: 23
   :language: python


Loading example data
^^^^^^^^^^^^^^^^^^^^

Next we load our Southern Ocean example data which was extracted from the CASCADE database :footcite:p:`deVries2024cascade` and then averaged with depth and time. When applying the pipeline to your own data, note that the data
needs to be in a `Pandas DataFrame format <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.

In addition to our predictors (`y_train`) we also need environmental data which match our predictions ('X_train'). 
This data was obtained from monthly climatologies from data sources such as the World Ocean Atlas :footcite:p:`Reagan2024`, NNGv2  :footcite:p:`Broullon2019, Broullon2020`.

.. literalinclude:: ../../../examples/2-phase.py
   :lines: 29-37
   :language: python

Training the model
^^^^^^^^^^^^^^^^^^

Next we train our model. Note that depending on the number of hyper-parameters specified in the
YAML file this can be computationally very expensive and it recommended to do this on a  :doc:`HPC system <hpc>`.  

.. literalinclude:: ../../../examples/2-phase.py
   :lines: 39-43
   :language: python

Making predictions
^^^^^^^^^^^^^^^^^^

After training our model we can make predictions on a new dataset (X_predict):

.. literalinclude:: ../../../examples/2-phase.py
   :lines: 45-52
   :language: python

Post-processing
^^^^^^^^^^^^^^^

Finally, we conduct the post-processing.

.. literalinclude:: ../../../examples/2-phase.py
   :lines: 55-62
   :language: python

Plotting
^^^^^^^^

Now that we have predictions we can plot them:

.. literalinclude:: ../../../examples/2-phase.py
   :lines: 64-105
   :language: python

.. figure:: ../../../examples/SO_predictions.png

Integrated inorganic carbon stock
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With abil.post can also estimate integrated inorganic carbon stock.
First we convert our abundances to cellular inorganic carbon, 
and then integrate accounting for grid volumes and unit conversions.

.. literalinclude:: ../../../examples/2-phase.py
   :lines: 107-120
   :language: python

Checking the output:

.. literalinclude:: ../../../examples/2-phase.py
   :lines: 122-130
   :language: python

.. code-block:: python

   >>> print(f"estimated integrated total: {mean:.2f} [{ci95_LL:.2f}, {ci95_UL:.2f}] Tg IC")
   estimated integrated total: 4.00 [0.69, 7.31] Tg IC

The resulting number is OK on a first order basis, but rather high, likely due to our example not accounting 
for the strong seasonal variation in biomass observed in this region.

References
^^^^^^^^^^
.. footbibliography::

