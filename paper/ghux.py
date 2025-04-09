"""
Gephyrocapsa huxleyi global distribution 
"""

# load dependencies
import pandas as pd
import xarray as xr
import numpy as np
from yaml import load
from yaml import CLoader as Loader
from abil.tune import tune
from abil.predict import predict
from abil.post import post
from abil.utils import upsample

import os
os.chdir('./paper')

# #load configuration yaml:
# with open('./data/2-phase.yml', 'r') as f:
#     model_config = load(f, Loader=Loader)

# #load training data:
# d = pd.read_csv("./data/training.csv")
# predictors = model_config['predictors']

# target = "Emiliania huxleyi HET"

# print(d)

# d = d.dropna(subset=predictors)
# d[target] = d[target].fillna(0)
# d = upsample(d, target, ratio=10)
# y = d[target]
# X_train = d[predictors]

# # #train your model:
# m = tune(X_train, y, model_config)
# m.train(model="rf")
# m.train(model="xgb")
# m.train(model="knn")

# #predict your model:
# X_predict = pd.read_csv("./data/env_mean_global_surface.csv")
# X_predict.set_index(["lat", "lon"], inplace=True)
# X_predict = X_predict[predictors]
# X_predict = X_predict.dropna(subset=predictors)

# m = predict(X_train, y, X_predict, model_config)
# m.make_prediction()

# # Posts
# targets = np.array([target])
# def do_post(statistic):
#     m = post(X_train, y, X_predict, model_config, statistic, datatype="abundance")
#     m.export_ds("my_first_2-phase_model")

# do_post(statistic="mean")
# do_post(statistic="ci95_UL")
# do_post(statistic="ci95_LL")

# plotting
import xarray as xr
import matplotlib.pyplot as plt
ds = xr.open_dataset("./ModelOutput/ghux_example/posts/my_first_2-phase_model_mean_abundance.nc")
print(ds)
ds['Emiliania huxleyi HET'].plot()

plt.show()