.. _yaml explained:

Model YAML Configuration
=========================

To define how your model is ran (e.g. where output is saved or which parameters to tune) a configuration YAML is used.

YAML is xyz


Paths:

>>> hpc: False
>>> local_root: /home/phyto/Abil/
>>> hpc_root: /user/work/ba18321/Abil/
>>> path_out: examples/ModelOutput/ #root + folder
>>> path_in:  examples/ModelOutput/mapie/ #root + folder
>>> env_data_path:  examples/data/prediction.csv #root + folder
>>> targets:  examples/data/targets.csv #root + folder
>>> training:  examples/data/training.csv

Predictors to be used:

>>> predictors: ["temperature", "din", "irradiance"]
    
System setup:    

>>> verbose: 1
>>> seed : 1 # random seed
>>> n_threads : 3 # how many cpu threads to use
>>> cv : 3
>>> predict_probability: False 


Ensemble configuration:

>>> ensemble_config: 
>>>   classifier: False
>>>   regressor: True
>>>   m1: "rf"
>>>   m2: "xgb"
>>>   m3: "knn"

Sampling and stratification:

>>> upsample: False
>>> stratify: True

Scoring method to be used:

>>> clf_scoring:
>>>  accuracy: balanced_accuracy
>>> reg_scoring:
>>>  R2: r2
>>>  MAE: neg_mean_absolute_error
>>>  RMSE: neg_root_mean_squared_error
>>>  tau: kendall_tau


