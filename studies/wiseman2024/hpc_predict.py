# import required packages
import pandas as pd
import numpy as np
import sys
from yaml import load
from yaml import CLoader as Loader
from abil.predict import predict
from sklearn.preprocessing import OneHotEncoder
from mapie.conformity_scores import GammaConformityScore

try:
    print(sys.argv[1])
    with open('/user/work/mv23682/Abil/wiseman2024/ensemble_regressor.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = True
    n_jobs = pd.to_numeric(sys.argv[1])
    n_spp = pd.to_numeric(sys.argv[2])
    root = model_config['hpc_root']
    model_config['cv'] = 10

except:
    with open('/home/mv23682/Documents/Abil/wiseman2024/ensemble_regressor.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = False
    n_jobs = 8
    n_spp = 1
    root = model_config['local_root']
    model_config['cv'] = 3

#define model config:
model_config['n_threads'] = n_jobs
targets = pd.read_csv(root + model_config['targets'])
d = pd.read_csv(root + model_config['training'])
target =  targets['Target'][n_spp]
d[target] = d[target].fillna(0)
d = d.dropna(subset=[target])
d = d.dropna()
print(target)
predictors = model_config['predictors']

X_predict =  pd.read_csv(root + model_config['env_data_path'])
X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
y = d[target]
X_train = d[predictors]

print("finished loading data")

m = predict(X_train, y, X_predict, model_config, n_jobs=n_jobs)
m.make_prediction(prediction_inference=True,conformity_score=GammaConformityScore(),alpha=[0.05])
