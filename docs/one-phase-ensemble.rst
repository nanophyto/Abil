1-phase Ensemble 
****************

YAML example
~~~~~~~~~~~~

Before running the model, model specifications need to be defined in a YAML file. 
For a detailed explanation of each parameter see :ref:`yaml explained`.

An example of YAML file of a 1-phase model is provided below.

.. literalinclude:: ../tests/regressor.yml
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
For instructions on how to install these packages, see :ref:`dependencies install`
and the Abil :ref:`install instructions`.

.. code-block:: python

    #load dependencies:
    import pandas as pd
    import numpy as np
    from yaml import load
    from yaml import CLoader as Loader
    from abil.tune import tune
    from abil.predict import predict
    from abil.post import post
    from abil.functions import example_data 
    import os, sys

Defining paths
^^^^^^^^^^^^^^

After loading the required packages we need to define our file paths.
Note that this is operating system specific, as Unix and Mac use '/' while for Windows '\' is used.

.. tab-set::

    .. tab-item:: Unix/MacOS

        .. code-block:: python

            #define root directory:
            os.chdir('/home/phyto-2/Abil/')  

            #load configuration yaml:
            with open('./tests/1-phase.yml', 'r') as f:
                model_config = load(f, Loader=Loader)


    .. tab-item:: Windows

        .. code-block:: python

            #define root directory:
            
            os.chdir('/home/phyto-2/Abil/')  

            #load configuration yaml:
            #with open('./tests/2-phase.yml', 'r') as f:
            #    model_config = load(f, Loader=Loader)

Creating example data
^^^^^^^^^^^^^^^^^^^^^

Next we create some example data. When applying the pipeline to your own data, note that the data
needs to be in a `Pandas DataFrame format <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.

.. code-block:: python

    #create some example training data:
    target_name =  "Emiliania huxleyi"

    X_train, X_predict, y = example_data(target_name, n_samples=1000, n_features=3, 
                                        noise=0.1, train_to_predict_ratio=0.7, 
                                        random_state=59)


Training the model
^^^^^^^^^^^^^^^^^^

Next we train our model. Note that depending on the number of hyper-parameters specified in the
YAML file this can be computationally very expensive and it recommended to do this on a HPC system. 


.. code-block:: python

    #train your model:
    m = tune(X_train, y, model_config)
    m.train(model="rf", classifier=True)

Making predictions
^^^^^^^^^^^^^^^^^^

After training our model we can make predictions on a new dataset (X_predict):

.. code-block:: python

    #predict your model:
    m = predict(X_train=X_train, y=y, X_predict=X_predict, 
        model_config=model_config, n_jobs=2)
    m.make_prediction()

Post-processing
^^^^^^^^^^^^^^^

Finally, we conduct the post-processing.

.. code-block:: python

    #post:
    m = post(model_config)
    m.export_ds("my_first_model")

    m = post(model_config)
    m.merge_performance(model="ens") 
    m.merge_performance(model="xgb")
    m.merge_performance(model="rf")
    m.merge_performance(model="knn")

    m.merge_parameters(model="rf")
    m.merge_parameters(model="xgb")
    m.merge_parameters(model="knn")
    m.estimate_carbon("pg poc")
    m.diversity()

    m.total()

    m.merge_env(X_predict)
    m.merge_obs("test",targets)

    m.export_ds("test")
    m.export_csv("test")

    vol_conversion = 1e3 #L-1 to m-3
    integ = m.integration(m, vol_conversion=vol_conversion)
    integ.integrated_totals(targets, monthly=True)
    integ.integrated_totals(targets)