"""
Regressor Ensemble 
"""
#paths:
import os
#handling data:
import numpy as np
import pandas as pd
import xarray as xr
from yaml import load
from yaml import CLoader as Loader
#abil functions:
from abil.tune import ModelTuner as tune
from abil.predict import ModelPredictor as predict
from abil.post import AbilPostProcessor as post
#plotting:
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(os.path.join(".", "examples"))

#load configuration yaml:
with open('regressor_BATS.yml', 'r') as f:
    model_config = load(f, Loader=Loader)

#load example training data:
d = pd.read_csv(os.path.join("data", "bats_training.csv"))
#define target:
target= "Gephyrocapsa huxleyi"
#define predictors based on YAML:
predictors = model_config['predictors']
#split training data in X_train and y
y = d[target]
X_train = d[predictors]

#plot training data:
#first we setup a helper function to plot the rasters:
def plot_depth_time(ax, da, variable=None, vmin=None, vmax=None,
             cbar_label=None):
    if isinstance(da, pd.DataFrame):
        da = da.to_xarray()
    if isinstance(da, xr.Dataset):
        da = da[variable]  
    m=da.plot(ax=ax,add_colorbar=True,
              vmin=vmin,vmax=vmax)
    m.colorbar.set_label(cbar_label)
    return ax
# then we can make the plots
fig, ax = plt.subplots(figsize=(14,5))
d.set_index(['depth', 'time'], inplace=True) # for plotting
plot_depth_time(ax, d, variable=target, cbar_label=r'abundance (cells L$^{-1}$)')
ax.set_title("Gephyrocapsa huxleyi abundance at BATS")
plt.savefig('BATS_observations.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.show()

#train your model:
m = tune(X_train, y, model_config)
m.train(model="rf")
m.train(model="xgb")
m.train(model="knn")

#load prediction data:
X_predict = pd.read_csv(os.path.join("data", "bats_prediction.csv"))
X_predict.set_index(['depth', 'time'], inplace=True)
X_predict = X_predict[predictors]

#plotting the prediction data
vars_list=[('temperature','Temperature','°C'),
           ('PAR', r'PAR ',r'W m$^{-2}$'),
           ('no3','Nitrate','µmol kg$^{-1}$'),
           ('TA','Total Alkalinity','µmol kg$^{-1}$')]
fig,axs=plt.subplots(4,1,figsize=(7,8),constrained_layout=True)
for ax,(v,title,unit),lab in zip(axs,vars_list,['A)','B)','C)', 'D)']):
    plot_depth_time(ax, X_predict, variable=v, vmin=None, vmax=None, cbar_label=unit)
    ax.set_title(rf'$\mathbf{{{lab}}}$ {title}',loc='left',fontsize=10,pad=6,y=1.02)
plt.savefig('BATS_environment.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.show()

#predict your model:
m = predict(X_train, y, X_predict, model_config)
m.make_prediction()

# Posts
targets = np.array([target])
def do_post(statistic):
    m = post(X_train, y, X_predict, model_config, statistic, datatype="abundance")
    m.export_ds("BATS")

do_post(statistic="mean")
do_post(statistic="ci95_UL")
do_post(statistic="ci95_LL")

# Plot the results
ds = xr.open_dataset(os.path.join("ModelOutput", "regressor", "posts", "BATS_mean_abundance.nc"))
ds_LL = xr.open_dataset(os.path.join("ModelOutput", "regressor", "posts", "BATS_ci95_LL_abundance.nc"))
ds_UL = xr.open_dataset(os.path.join("ModelOutput", "regressor", "posts", "BATS_ci95_UL_abundance.nc"))

fig,axs=plt.subplots(3,1,figsize=(7,6),constrained_layout=True)
for ax,da,title,lab in [
    (axs[0], ds[target], 'Mean Abundance','A)'),
    (axs[1], ds_LL[target], '95% CI Lower Limit','B)'),
    (axs[2], ds_UL[target], '95% CI Upper Limit','C)')]:
    plot_depth_time(ax, da, vmin=0, cbar_label=r'abundance (cells L$^{-1}$)')
    ax.set_title(rf'$\mathbf{{{lab}}}$ {title}')
plt.savefig('BATS_predictions.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.show()

# Check statistics
obs_abundance = pd.read_csv(os.path.join("ModelOutput", "regressor", "posts", 
                                "performance", "ens_performance.csv"))
# print
print(
    f"error metrics:\n"
    f"R2 = {obs_abundance['R2'][0]:.2f}\n"
    f"relative RMSE = {obs_abundance['rRMSE'][0]*100:.2f} (%)\n"
    f"relative MAE = {obs_abundance['rMAE'][0]*100:.2f} (%)"
)

# Merge obs and model for plotting
targets = np.array([target])
def do_stats(statistic):
    m = post(X_train, y, X_predict, model_config, statistic, datatype="abundance")
    m.merge_obs(file_name="BATS", index_cols=['depth', 'time']) #default is lat, lon, depth, time

do_stats(statistic="mean")

obs_abundance = pd.read_csv(os.path.join("ModelOutput", "regressor", "posts", 
                                "BATS_obs_abundance.csv"))

# plot
sns.regplot(data = obs_abundance, x = 'Gephyrocapsa huxleyi', 
           y= 'Gephyrocapsa huxleyi_mod')
plt.xlabel("obs")
plt.ylabel("model")
plt.title('Observed vs modelled abundances')
plt.savefig('BATS_obs_preds.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.show()
