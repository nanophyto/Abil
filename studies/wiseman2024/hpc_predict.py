# import required packages
import pandas as pd
import sys
from yaml import load
from yaml import CLoader as Loader
from abil.predict import predict

try:
    print(sys.argv[1])
    with open('/user/work/mv23682/Abil/studies/wiseman2024/ensemble_regressor.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = True
    n_jobs = pd.to_numeric(sys.argv[1])
    n_spp = pd.to_numeric(sys.argv[2])
    root = model_config['root']
    model_config['cv'] = 10

except:
    with open('/home/mv23682/Documents/Abil/studies/wiseman2024/ensemble_regressor.yml', 'r') as f:
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
d = d.dropna(subset=[target])
print(target)
predictors = model_config['predictors']
d = d.dropna(subset=predictors)

X_predict =  pd.read_csv(root + model_config['prediction'])
X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
X_predict = X_predict.dropna()
y = d[target]
X_train = d[predictors]

print("finished loading data")

m = predict(X_train, y, X_predict, model_config, n_jobs=n_jobs)
m.make_prediction()
