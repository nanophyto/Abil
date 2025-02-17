# import required packages
import pandas as pd
import numpy as np
import sys
from yaml import load
from yaml import CLoader as Loader
from abil.tune import tune 
from abil.functions import upsample
from sklearn.preprocessing import OneHotEncoder

with open('/user/work/ba18321/Abil/studies/devries2024/2-phase.yml', 'r') as f:
    model_config = load(f, Loader=Loader)

model_config['hpc'] = True
n_jobs = pd.to_numeric(sys.argv[1])
n_spp = pd.to_numeric(sys.argv[2])
root = model_config['root']
model_config['cv'] = 10
model = sys.argv[3]
predictors = model_config['predictors']

#define model config:
model_config['n_threads'] = n_jobs
targets = pd.read_csv(root + model_config['targets'])
d = pd.read_csv(root + model_config['training'])
target =  targets['Target'][n_spp]
d[target] = d[target].fillna(0)
d = d.dropna(subset=[target])
d = d.dropna(subset=predictors)
d = upsample(d, target, ratio=10)

y = d[target]
X_train = d[predictors]

#setup model:
m = tune(X_train, y, model_config)
#run model:
m.train(model=model, regressor=True, classifier=True, log="both")
