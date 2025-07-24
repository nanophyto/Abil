# import required packages
import pandas as pd
import sys, os
from yaml import load
from yaml import CLoader as Loader
from abil.post import post
from datetime import datetime

current_date = datetime.today().strftime('%Y-%m-%d')

print(sys.argv[0])
dirpath = os.path.dirname(os.path.abspath(__file__))
conffile = os.path.abspath(os.path.join(dirpath,'ensemble_regressor.yml'))
root = os.path.abspath(os.path.join(dirpath,'..','..'))

#load model arguments
with open(conffile, 'r') as f:
    model_config = load(f, Loader=Loader)

#define model config:
model_config['hpc'] = True

#load data
targets = pd.read_csv(os.path.join(root,model_config['targets']))
target =  targets['Target'][0]
targets = targets['Target'].values
target_subset = targets[:2] # subset for estimate_applicability and merge_obs
d = pd.read_csv(os.path.join(root,model_config['training']))
predictors = model_config['predictors']
d = d.dropna(subset=predictors)

X_predict =  pd.read_csv(os.path.join(root,model_config['prediction']))
X_predict.set_index(['time','depth','lat','lon'],inplace=True)
X_predict = X_predict[model_config['predictors']]
X_predict = X_predict.dropna()

y = d[target]
X_train = d[predictors]

# Create a mask where y is a pseudo-zero
mask = y != 0
y = y[mask]
X_train = X_train[mask]

def do_post(statistic):
    m = post(X_train, y, X_predict, model_config, statistic)
    
    if statistic == "mean":
        print('begin estimate_applicability')
        m.estimate_applicability(target_subset)

        print('begin merge_obs')
        m.merge_obs(current_date,target_subset)
    
        print('begin process_resampled_runs')
        m.process_resampled_runs()

    print('begin export_ds')
    m.export_ds(current_date)

    print('begin integration')
    magnitude_conversion = 1e-21
    molar_mass = 12.01
    integ = m.integration(m, magnitude_conversion=magnitude_conversion,molar_mass=molar_mass,rate=True)
    if statistic == "mean":
        integ.integrated_totals(targets)
    else:
        integ.integrated_totals(target_subset)

    print('do_post for: ', statistic, ' complete')

do_post(statistic="mean")
do_post(statistic="ci95_UL")
do_post(statistic="ci95_LL")

