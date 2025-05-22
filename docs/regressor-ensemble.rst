Regressor YAML example
*****************************

.. literalinclude:: ../tests/regressor.yml
   :language: yaml

Regressor Ensemble code example
*****************************
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
            os.chdir('/home/phyto-2/Abil/')  

            #load configuration yaml:
            with open('./tests/regressor.yml', 'r') as f:
                model_config = load(f, Loader=Loader)

            #create some example training data:
            target_name =  "Emiliania huxleyi"

            X_train, X_predict, y = example_data(target_name, n_samples=1000, n_features=3, 
                                                noise=0.1, train_to_predict_ratio=0.7, 
                                                random_state=59)

            #train your model:
            m = tune(X_train, y, model_config)
            m.train(model="rf", regressor=True)

            #predict your model:
            m = predict(X_train=X_train, y=y, X_predict=X_predict, 
                model_config=model_config, n_jobs=2)
            m.make_prediction()

            #post:
            m = post(model_config)
            m.export_ds("my_first_model")


    .. tab-item:: Windows


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
            
            #os.chdir('/home/phyto-2/Abil/')  

            #load configuration yaml:
            #with open('./tests/2-phase.yml', 'r') as f:
            #    model_config = load(f, Loader=Loader)

            #create some example training data:
            target_name =  "Emiliania huxleyi"

            X_train, X_predict, y = example_data(target_name, n_samples=1000, n_features=3, 
                                                noise=0.1, train_to_predict_ratio=0.7, 
                                                random_state=59)

            #train your model:
            m = tune(X_train, y, model_config)
            m.train(model="rf", classifier=True)

            #predict your model:
            m = predict(X_train=X_train, y=y, X_predict=X_predict, 
                model_config=model_config, n_jobs=2)
            m.make_prediction()

            #post:
            m = post(model_config)
            m.export_ds("my_first_model")
