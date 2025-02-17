# import required packages
import pandas as pd
import sys
from yaml import load
from yaml import CLoader as Loader
from abil.predict import predict
from abil.functions import upsample

with open('/user/work/ba18321/Abil/studies/devries2024/2-phase.yml', 'r') as f:
    model_config = load(f, Loader=Loader)
model_config['hpc'] = True
n_jobs = pd.to_numeric(sys.argv[1])
n_spp = pd.to_numeric(sys.argv[2])
root = model_config['root']
model_config['cv'] = 10

#define model config:
model_config['n_threads'] = n_jobs
targets = pd.read_csv(root + model_config['targets'])
d = pd.read_csv(root + model_config['training'])
target =  targets['Target'][n_spp]
predictors = model_config['predictors']
d[target] = d[target].fillna(0)
d = d.dropna(subset=[target])
d = d.dropna(subset=predictors)
d = upsample(d, target, ratio=10)
print(target)

X_predict =  pd.read_csv(root + model_config['prediction'])
X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
y = d[target]
X_train = d[predictors]

print("finished loading data")

m = predict(X_train, y, X_predict, model_config, n_jobs=n_jobs)

m.make_prediction()