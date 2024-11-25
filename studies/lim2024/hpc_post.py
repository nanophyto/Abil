# import required packages
import sys
from yaml import load
from yaml import CLoader as Loader
from abil.post import post
import pandas as pd

from datetime import datetime
current_date = datetime.today().strftime('%Y-%m-%d')


try:
    print(sys.argv[0])
    with open('/user/work/mv23682/Abil/studies/wiseman2024/ensemble_regressor.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = True
    root = model_config['hpc_root']

except:
    with open('/home/mv23682/Documents/Abil/studies/wiseman2024/ensemble_regressor.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = False
    root = model_config['local_root']

print("path:")
print(root+model_config['targets'])
targets=pd.read_csv(root+model_config['targets'])
targets=targets['Target'].values

X_predict = pd.read_csv(root + model_config['prediction'])
X_predict.set_index(['time','depth','lat','lon'],inplace=True)
X_predict = X_predict[model_config['predictors']]

def do_post(pi):
    m = post(model_config, pi=pi)
    m.merge_performance(model="ens") 
    m.merge_performance(model="xgb")
    m.merge_performance(model="rf")
    m.merge_performance(model="knn")

    m.merge_parameters(model="rf")
    m.merge_parameters(model="xgb")
    m.merge_parameters(model="knn")

    m.merge_env(X_predict)
    m.merge_obs(current_date,targets)

    m.export_ds(current_date)
    m.export_csv(current_date)

    magnitude_conversion = 1e-21
    molar_mass = 12.01
    integ = m.integration(m, magnitude_conversion=magnitude_conversion,molar_mass=molar_mass,rate=True)
    print(targets)
    integ.integrated_totals(targets)
    integ.integrated_totals(targets, subset_depth=100)

do_post(pi="50")
do_post(pi="95_UL")
do_post(pi="95_LL")

