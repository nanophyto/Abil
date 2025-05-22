"""
Gephyrocapsa huxleyi global distribution
"""
# load dependencies
import pandas as pd
import xarray as xr
import numpy as np
from yaml import load
from yaml import CLoader as Loader
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from abil.tune import tune
from abil.predict import predict
from abil.post import post
from abil.utils import upsample

import os
os.chdir('./paper')

# Load configuration yaml:
with open('./data/2-phase.yml', 'r') as f:
    model_config = load(f, Loader=Loader)

# Load training data:
d = pd.read_csv("./data/training.csv")
predictors = model_config['predictors']
target = "Gephyrocapsa huxleyi HET"
d = d.dropna(subset=predictors)
d[target] = d[target].fillna(0)
d = upsample(d, target, ratio=10)
y = d[target]
X_train = d[predictors]

# Train your model:
m = tune(X_train, y, model_config)
m.train(model="rf")
m.train(model="xgb")
m.train(model="knn")

# Predict your model:
X_predict = pd.read_csv("./data/env_mean_global_surface.csv")
X_predict.set_index(["lat", "lon"], inplace=True)
X_predict = X_predict[predictors]
X_predict = X_predict.dropna(subset=predictors)

m = predict(X_train, y, X_predict, model_config)
m.make_prediction()

# Posts
targets = np.array([target])
def do_post(statistic):
    m = post(X_train, y, X_predict, model_config, statistic, datatype="abundance")
    m.export_ds("my_first_2-phase_model")

do_post(statistic="mean")
do_post(statistic="ci95_UL")
do_post(statistic="ci95_LL")

# Load the data
ds = xr.open_dataset("./ModelOutput/ghux_example/posts/my_first_2-phase_model_mean_abundance.nc")
ds_UL = xr.open_dataset("./ModelOutput/ghux_example/posts/my_first_2-phase_model_ci95_UL_abundance.nc")
ds_LL = xr.open_dataset("./ModelOutput/ghux_example/posts/my_first_2-phase_model_ci95_LL_abundance.nc")
d = pd.read_csv("./data/training.csv")

# Log-transform data
ds['Gephyrocapsa huxleyi HET'] = np.log10(ds['Gephyrocapsa huxleyi HET'] + 1)
ds_UL['Gephyrocapsa huxleyi HET'] = np.log10(ds_UL['Gephyrocapsa huxleyi HET'] + 1)
ds_LL['Gephyrocapsa huxleyi HET'] = np.log10(ds_LL['Gephyrocapsa huxleyi HET'] + 1)
d['Gephyrocapsa huxleyi HET'] = np.log10(d['Gephyrocapsa huxleyi HET'])

# Create figure with adjusted subplot spacing
fig, axs = plt.subplots(2, 2, figsize=(12, 6),
                       subplot_kw={'projection': ccrs.PlateCarree()})
axs = axs.flatten()

# Titles and panel labels
titles = ['Training Data', 'Mean Abundance', '95% CI Lower Limit', '95% CI Upper Limit']
panel_labels = ['A)', 'B)', 'C)', 'D)']

# Custom title function to handle labels
def add_title(ax, title, label, y=1.1):
    ax.set_title(f'$\mathbf{{{label}}}$  {title}', loc='left', pad=10, y=y, fontsize=10)
    return ax

# Plot Training Data
sc = axs[0].scatter(d['lon'], d['lat'], c=d['Gephyrocapsa huxleyi HET'],
                   cmap='viridis', s=10, transform=ccrs.PlateCarree(),
                   vmin=0)
add_title(axs[0], titles[0], panel_labels[0])
axs[0].set_ylim([-90, 90])
cbar0 = plt.colorbar(sc, ax=axs[0], shrink=0.6, pad=0.1)
cbar0.ax.tick_params(labelsize=8)
cbar0.set_label('log$_{10}$ abundance (cells L$^{-1}$)', size=8)

# Plot Mean Abundance
p1 = ds['Gephyrocapsa huxleyi HET'].plot(ax=axs[1], cmap='viridis', add_colorbar=False,
                                    vmin=0)
add_title(axs[1], titles[1], panel_labels[1])
axs[1].set_ylim([-90, 90])
cbar1 = plt.colorbar(p1, ax=axs[1], shrink=0.6, pad=0.1)
cbar1.ax.tick_params(labelsize=8)
cbar1.set_label('log$_{10}$ abundance (cells L$^{-1}$)', size=8)

# Plot CI Lower Limit
p2 = ds_LL['Gephyrocapsa huxleyi HET'].plot(ax=axs[2], cmap='viridis', add_colorbar=False,
                                       vmin=0)
add_title(axs[2], titles[2], panel_labels[2])
axs[2].set_ylim([-90, 90])
cbar2 = plt.colorbar(p2, ax=axs[2], shrink=0.6, pad=0.1)
cbar2.ax.tick_params(labelsize=8)
cbar2.set_label('log$_{10}$ abundance (cells L$^{-1}$)', size=8)

# Plot CI Upper Limit
p3 = ds_UL['Gephyrocapsa huxleyi HET'].plot(ax=axs[3], cmap='viridis', add_colorbar=False,
                                       vmin=0)
add_title(axs[3], titles[3], panel_labels[3])
axs[3].set_ylim([-90, 90])
cbar3 = plt.colorbar(p3, ax=axs[3], shrink=0.6, pad=0.1)
cbar3.ax.tick_params(labelsize=8)
cbar3.set_label('log$_{10}$ abundance (cells L$^{-1}$)', size=8)

# Add coastlines
for ax in axs:
    ax.add_feature(cfeature.LAND, color='gray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

plt.tight_layout()
plt.savefig('figure_1.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
