Quick Start
============

1-phase random forest regressor
***********************

load dependencies:


set your directory:

.. tab-set::

    .. tab-item:: Linux/MacOS

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

            #define root directory:
            os.chdir('/home/phyto/Abil/')  

            #load configuration yaml:
            with open('./examples/configuration/2-phase.yml', 'r') as f:
                model_config = load(f, Loader=Loader)

            #load the training data:
            d = pd.read_csv(model_config['local_root'] + model_config['training'])

            #define your environmental predictors 
            #(note that these should be found in your training.csv!):
            predictors = ["temperature", "din", "irradiance"]

            #define your target value (in this case the species *E. huxleyi*)
            target =  "Emiliania huxleyi"

            #drop any missing values:
            d = d.dropna(subset=[target])
            d = d.dropna(subset=predictors)

            #define your X and y:
            X_train = d[predictors]
            y = d[target]

            #train your model:
            m = tune(X, y, model_config)
            m.train(model="rf", classifier=True)

            #predict your model:
            X_predict = pd.read_csv("./examples/data/prediction.csv")
            X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
            m = predict(X_train=X_train, y=y, X_predict=X_predict, 
                model_config=model_config, n_jobs=2)
            m.make_prediction(prediction_inference=True)

            #post:
            m = post(model_config)
            m.export_ds("my_first_model")




    .. tab-item:: Windows

        .. code-block:: python

            import os, sys
            os.chdir('\home\phyto\Abil\') 
