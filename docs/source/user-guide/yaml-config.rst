.. _yaml_config:

Model YAML Configuration
=========================

To define how your model is ran (e.g. where output is saved or which parameters to tune) a configuration YAML is used.

Paths:
------

>>> root: ./
>>> run_name: run_name #update for specific run name
>>> path_out: studies/study_name/ModelOutput/ #root + folder
>>> prediction:  studies/study_name/data/prediction.csv #root + folder
>>> targets:  studies/study_name/data/targets.csv #root + folder
>>> training:  studies/study_name/data/training.csv #root + folder

Predictors to be used:
----------------------
>>> predictors: ["temperature", "din", "irradiance"]
    
System setup:    
-------------
>>> verbose: 1 # scikit-learn warning verbosity
>>> seed : 1 # random seed
>>> n_threads : 1 # how many cpu threads to use
>>> cv : 3 # number of cross-folds


Ensemble configuration:
------------------------
>>> ensemble_config: 
>>>   classifier: False #set as True for 2-phase model (classifier only not supported)
>>>   regressor: True #set as True for regressor model or 2-phase model
>>>   m1: "rf"
>>>   m2: "xgb"
>>>   m3: "knn"

Sampling and stratification:
----------------------------
>>> upsample: False
>>> stratify: True #zero stratification


