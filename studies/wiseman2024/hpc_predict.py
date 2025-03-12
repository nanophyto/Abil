# import required packages
import pandas as pd
import sys, os
from yaml import load
from yaml import CLoader as Loader
from abil.predict import predict

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
model_config['n_threads'] = n_jobs
model_config['cv'] = 10

#load data
targets = pd.read_csv(os.path.join(root,model_config['targets']))
d = pd.read_csv(os.path.join(root,model_config['training']))
target =  targets['Target'][n_spp]
predictors = model_config['predictors']
d = d.dropna(subset=[target])
d = d.dropna(subset=predictors)

X_predict =  pd.read_csv(os.path.join(root,model_config['prediction']))
X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
X_predict = X_predict.dropna()

y = d[target]
X_train = d[predictors]

print("finished loading data")

#setup model
m = predict(X_train, y, X_predict, model_config, n_jobs=n_jobs)

#run model
m.make_prediction()
