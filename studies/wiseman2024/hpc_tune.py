# import required packages
import pandas as pd
import sys, os
from yaml import load
from yaml import CLoader as Loader
from abil.tune import tune

#define directories
print(sys.argv[1])
dirpath = os.path.dirname(os.path.abspath(__file__))
conffile = os.path.abspath(os.path.join(dirpath,'ensemble_regressor.yml'))
root = os.path.abspath(os.path.join(dirpath,'..','..'))

#load model arguments
with open(conffile, 'r') as f:
    model_config = load(f, Loader=Loader)

#define model config:
model_config['hpc'] = True
n_jobs = pd.to_numeric(sys.argv[1])
n_spp = pd.to_numeric(sys.argv[2])
model = sys.argv[3]
model_config['n_threads'] = n_jobs
model_config['cv'] = 10

#load data
targets = pd.read_csv(os.path.join(root,model_config['targets']))
d = pd.read_csv(os.path.join(root,model_config['training']))
target =  targets['Target'][n_spp]
predictors = model_config['predictors']
d = d.dropna(subset=[target])
d = d.dropna(subset=predictors)

y = d[target]
X_train = d[predictors]

#setup model:
m = tune(X_train, y, model_config)

#run model:
m.train(model=model, regressor=True)
