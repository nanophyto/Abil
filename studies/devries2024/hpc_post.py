# import required packages
import sys
from yaml import load
from yaml import CLoader as Loader
from abil.post import post
import pandas as pd

with open('/user/work/ba18321/Abil/studies/devries2024/2-phase.yml', 'r') as f:
    model_config = load(f, Loader=Loader)
root = model_config['root']

file_name = model_config['run_name']

print("path:")
print(root + model_config['targets'])

targets = pd.read_csv(root + model_config['targets'])
targets =  targets['Target'].values

X_predict = pd.read_csv(root + model_config['prediction'])
X_predict.set_index(['time','depth','lat','lon'],inplace=True)
X_predict = X_predict[model_config['predictors']]

def do_post(pi, datatype, diversity=False):
    m = post(model_config, pi=pi)
    m.estimate_carbon(datatype)
    if diversity:
        m.diversity()

    m.total()
    m.merge_env(X_predict)
    m.merge_obs(file_name, targets)

    m.export_ds(file_name)
    m.export_csv(file_name)

    vol_conversion = 1e3 #L-1 to m-3
    integ = m.integration(m, vol_conversion=vol_conversion)
    integ.integrated_totals(targets, monthly=True)
    integ.integrated_totals(targets)

do_post(pi="50", datatype="pg poc", diversity=True)
do_post(pi="50", datatype="pg pic")