1-phase Ensemble 
****************

In this example we will use Abil to predict the biomass of a highly abundant calcifying nanoplankton which is important for the carbon cycle (`Gephyrocapsa huxleyi` HET).


YAML example
~~~~~~~~~~~~

Before running the model, model specifications need to be defined in a YAML file. 
For a detailed explanation of each parameter see :ref:`yaml_config`.

An example of YAML file of a 1-phase model is provided below.

.. literalinclude:: ../../examples/regressor.yml
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

.. literalinclude:: ../../examples/regressor.py
   :lines: 4-20
   :language: python

Loading the configuration YAML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After loading the required packages we need to define our file paths.

.. literalinclude:: ../../examples/regressor.py
   :lines: 22
   :language: python


Loading example data
^^^^^^^^^^^^^^^^^^^^^

Next we load some example data, here we utilize abundance data from the CASCADE database (10.5281/zenodo.12797197).
The CASCADE database provides observations gridded to 1 degree x 1 degree x 5 meters x 1 month. 
For the example we focus on the Southern Ocean, a region with high Gephyrocapsa huxleyi abundances.
Furthermore, we averaged the observations with time and use only observations in the top 5 meters to speed up the simulations. 
In addition to our predictors (`y_train`) we also need environmental data which match our predictions ('X_train'). 
This data was obtained from monthly climatologies from data sources such as the World Ocean Atlas, NNGv2, and Castant et al.
When applying the pipeline to your own data, note that the data
needs to be in a `Pandas DataFrame format <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.

.. literalinclude:: ../../examples/regressor.py
   :lines: 24-35
   :language: python

Training the model
^^^^^^^^^^^^^^^^^^

Next we train our model. Note that depending on the number of hyper-parameters specified in the
YAML file this can be computationally very expensive and it recommended to do this on a HPC system. 

.. literalinclude:: ../../examples/regressor.py
   :lines: 37-41
   :language: python

Making predictions
^^^^^^^^^^^^^^^^^^

After training our model we can make predictions on the Southern Ocean dataset:

First we need to load our environmental data to make the predictions on (X_predict):

.. literalinclude:: ../../examples/regressor.py
   :lines: 43-44
   :language: python

Then we can make our predictions:

.. literalinclude:: ../../examples/regressor.py
   :lines: 47-49
   :language: python

Post-processing
^^^^^^^^^^^^^^^

Finally, we conduct the post-processing.

.. literalinclude:: ../../examples/regressor.py
   :lines: 51-60
   :language: python

Plotting
^^^^^^^^

Now that we have predictions we can plot them:

.. literalinclude:: ../../examples/regressor.py
   :lines: 62-139
   :language: python

.. figure:: ../../examples/figure_1.png