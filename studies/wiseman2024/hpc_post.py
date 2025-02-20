# import required packages
import pandas as pd
import sys
from yaml import load
from yaml import CLoader as Loader
from abil.post import post


from datetime import datetime
current_date = datetime.today().strftime('%Y-%m-%d')


try:
    print(sys.argv[0])
    with open('/user/work/mv23682/Abil/studies/wiseman2024/ensemble_regressor.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = True
    root = model_config['root']

except:
    with open('/home/mv23682/Documents/Abil/studies/wiseman2024/ensemble_regressor.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = False
    root = model_config['local_root']

targets = pd.read_csv(root+model_config['targets'])
target =  targets['Target'][0]
targets = targets['Target'].values

d = pd.read_csv(root + model_config['training'])
predictors = model_config['predictors']
d = d.dropna(subset=predictors)

X_predict = pd.read_csv(root + model_config['prediction'])
X_predict.set_index(['time','depth','lat','lon'],inplace=True)
X_predict = X_predict[model_config['predictors']]
X_predict = X_predict.dropna()
y = d[target]
X_train = d[predictors]

def do_post(statistic):
    m = post(X_train, y, X_predict, model_config, statistic)
    m.estimate_applicability()
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
    integ.integrated_totals(targets)

do_post(statistic="mean")
do_post(statistic="median")
do_post(statistic="sd")
do_post(statistic="ci95_UL")
do_post(statistic="ci95_LL")

