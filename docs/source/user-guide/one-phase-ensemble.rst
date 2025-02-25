1-phase Ensemble 
****************

YAML example
~~~~~~~~~~~~

Before running the model, model specifications need to be defined in a YAML file. 
For a detailed explanation of each parameter see :ref:`yaml_config`.

An example of YAML file of a 1-phase model is provided below.

.. literalinclude:: ../../../tests/regressor.yml
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
   :lines: 4-10
   :language: python

Loading the configuration YAML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After loading the required packages we need to define our file paths.
Note that this is operating system specific, as Unix and Mac use '/' while for Windows '\' is used.

.. literalinclude:: ../../examples/regressor.py
   :lines: 16-17
   :language: python


Creating example data
^^^^^^^^^^^^^^^^^^^^^

Next we create some example data. When applying the pipeline to your own data, note that the data
needs to be in a `Pandas DataFrame format <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.

.. literalinclude:: ../../examples/regressor.py
   :lines: 20-23
   :language: python

Training the model
^^^^^^^^^^^^^^^^^^

Next we train our model. Note that depending on the number of hyper-parameters specified in the
YAML file this can be computationally very expensive and it recommended to do this on a HPC system. 

.. literalinclude:: ../../examples/regressor.py
   :lines: 26-29
   :language: python

Making predictions
^^^^^^^^^^^^^^^^^^

After training our model we can make predictions on a new dataset (X_predict):

.. literalinclude:: ../../examples/regressor.py
   :lines: 32-33
   :language: python

Post-processing
^^^^^^^^^^^^^^^

Finally, we conduct the post-processing.

.. literalinclude:: ../../examples/regressor.py
   :lines: 36-58
   :language: python