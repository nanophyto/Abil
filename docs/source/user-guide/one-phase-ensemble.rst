1-phase Ensemble 
****************

In this example we will use Abil to predict the biomass of the calcifying nanoplankton *Gephyrocapsa huxleyi*.
This species is highly abundant in the global oceans, and plays an important role in the carbon cycle.
We fill focus this example on the Bermuda Atlantic Time Series (BATS), a location with great temporal coverage of *G. huxleyi*.
In this location, wind-driven mixing in the winter drives nutrient upwelling which results in high concentrations of *G. huxleyi*.
In the summer, as nutrient influx decreases due to lower mixing and subsequently the abundance of *G. huxleyi* decreases. 
An interesting temporal dynamic we will try to capture with our machine learning ensemble. 

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
   :lines: 4-17
   :language: python

Loading the configuration YAML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After loading the required packages we need to define our file paths.

.. literalinclude:: ../../examples/regressor.py
   :lines: 19
   :language: python

Then we can load the YAML:

.. literalinclude:: ../../examples/regressor.py
   :lines: 21-23
   :language: python

Loading example data
^^^^^^^^^^^^^^^^^^^^^

Next we load some example data, here we utilize abundance data from the CASCADE database (10.5281/zenodo.12797197).
The CASCADE database provides observations gridded to 1 degree x 1 degree x 5 meters x 1 month. 
For the example we have subset the data for Bermuda and have averaged the observations with latitude and longitude. 
In addition to our predictors (`y_train`) we also need environmental data which match our predictions ('X_train'). 
This data was obtained from monthly climatologies from data sources such as the World Ocean Atlas, and NNGv2.
When applying the pipeline to your own data, note that the data
needs to be in a `Pandas DataFrame format <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.

.. literalinclude:: ../../examples/regressor.py
   :lines: 24-32
   :language: python

Plotting example data
^^^^^^^^^^^^^^^^^^^^^

We can have a look at the example data. It already has pretty good data coverage.

.. literalinclude:: ../../examples/regressor.py
   :lines: 34-52
   :language: python

.. figure:: ../../examples/BATS_observations.png

Training the model
^^^^^^^^^^^^^^^^^^

Next we train our model. Note that depending on the number of hyper-parameters specified in the
YAML file this can be computationally very expensive and it recommended to do this on a HPC system. 

.. literalinclude:: ../../examples/regressor.py
   :lines: 54-58
   :language: python



Making predictions
^^^^^^^^^^^^^^^^^^

After training our model we can make predictions on the BATS environmental dataset:

First we need to load our environmental data to make the predictions on (X_predict):

.. literalinclude:: ../../examples/regressor.py
   :lines: 60-63
   :language: python

We can also quickly plot the environmental data. Note the seasonality in key parameters such as PAR and temperature.


.. literalinclude:: ../../examples/regressor.py
   :lines: 65-72
   :language: python

.. figure:: ../../examples/BATS_environment.png


Then we can make our predictions:

.. literalinclude:: ../../examples/regressor.py
   :lines: 74-76
   :language: python

Post-processing
^^^^^^^^^^^^^^^

Finally, we conduct the post-processing.

.. literalinclude:: ../../examples/regressor.py
   :lines: 79-86
   :language: python

Plotting
^^^^^^^^

Now that we have predictions we can plot them:

.. literalinclude:: ../../examples/regressor.py
   :lines: 89-100
   :language: python

.. figure:: ../../examples/BATS_predictions.png