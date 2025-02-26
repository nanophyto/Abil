"""
2-phase Ensemble 
"""
import numpy as np
from yaml import load
from yaml import CLoader as Loader
from abil.tune import tune
from abil.predict import predict
from abil.post import post
from abil.utils import example_data 

import os
os.chdir('./docs/examples/')

#load configuration yaml:
with open('2-phase.yml', 'r') as f:
    model_config = load(f, Loader=Loader)

#create some example training data:
target_name =  "Emiliania huxleyi"
X_train, X_predict, y = example_data(target_name, n_samples=1000, n_features=3, 
                                    noise=0.1, train_to_predict_ratio=0.7, 
                                    random_state=59)

#train your model:
m = tune(X_train, y, model_config)
m.train(model="rf")
m.train(model="xgb")
m.train(model="knn")

#predict your model:
m = predict(X_train, y, X_predict, model_config)
m.make_prediction()

# Posts
targets = np.array([target_name])
def do_post(statistic):
    m = post(X_train, y, X_predict, model_config, statistic, datatype="poc")
    
    m.estimate_applicability()
    m.estimate_carbon("pg poc")
    m.total()

    m.merge_env()
    m.merge_obs("predictions_obs", targets)

    m.export_ds("my_first_2-phase_model")

    vol_conversion = 1e3 #L-1 to m-3
    integ = m.integration(m, vol_conversion=vol_conversion)
    integ.integrated_totals(targets, monthly=True)
    integ.integrated_totals(targets)

do_post(statistic="mean")
do_post(statistic="median")
do_post(statistic="std")
do_post(statistic="ci95_UL")
do_post(statistic="ci95_LL")