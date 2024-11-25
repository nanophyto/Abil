# import required packages
import pandas as pd
import numpy as np
import sys
from yaml import load
from yaml import CLoader as Loader
from abil.tune import tune 
from sklearn.preprocessing import OneHotEncoder

try:
    print(sys.argv[1])
    with open('/user/work/mv23682/Abil/wiseman2024/studies/ensemble_regressor.yml', 'r') as f:
        model_config = load(f, Loader=Loader)

    model_config['hpc'] = True
    n_jobs = pd.to_numeric(sys.argv[1])
    n_spp = pd.to_numeric(sys.argv[2])
    root = model_config['hpc_root']
    model_config['cv'] = 10
    model = sys.argv[3]
    predictors = model_config['predictors']

except:
    with open('/home/mv23682/Documents/Abil/wiseman2024/studies/ensemble_regressor.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = False
    n_jobs = 8
    n_spp = 1
    root = model_config['local_root']
    model_config['cv'] = 3
    
    with open('/home/mv23682/Documents/Abil/wiseman2024/studies/ensemble_regressor.yml', 'r') as f:
        model_config_local = load(f, Loader=Loader)    
    
    model_config['param_grid'] = model_config_local['param_grid'] 
    model = "rf"


#define model config:
model_config['n_threads'] = n_jobs
targets = pd.read_csv(root + model_config['targets'])
d = pd.read_csv(root + model_config['training'])
target =  targets['Target'][n_spp]
d = d.dropna(subset=[target])
d = d.dropna(subset=predictors)

y = d[target]
X_train = d[predictors]

#setup model:
m = tune(X_train, y, model_config, regions=None)
#run model:
m.train(model=model, regressor=True)
