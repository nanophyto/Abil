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

d = pd.read_csv(root + model_config['training'])
predictors = model_config['predictors']

y = d[targets]
X_train = d[predictors]

def do_post(statistic, datatype, diversity=False):
    m = post(X_train,y, X_predict, model_config, statistic, datatype="poc")

    m.estimate_applicability()

    m.estimate_carbon(datatype)

    if diversity:
        m.diversity()

    m.total()
    m.merge_env()
    m.merge_obs("test",targets)

    m.export_ds("test")

    vol_conversion = 1e3 #L-1 to m-3
    integ = m.integration(m, vol_conversion=vol_conversion)
    integ.integrated_totals(targets, monthly=True)


do_post(statistic="mean", datatype="pg poc", diversity=True)
do_post(statistic="median", datatype="pg poc", diversity=True)
do_post(statistic="sd", datatype="pg poc", diversity=True)
do_post(statistic="ci95_UL", datatype="pg poc", diversity=True)
do_post(statistic="ci95_LL", datatype="pg poc", diversity=True)
    
do_post(statistic="mean", datatype="pg pic")
do_post(statistic="median", datatype="pg pic")
do_post(statistic="sd", datatype="pg pic")
do_post(statistic="ci95_UL", datatype="pg pic")
do_post(statistic="ci95_LL", datatype="pg pic")