"""
2-phase Ensemble 
"""
#handling data:
import numpy as np
import pandas as pd
import xarray as xr
from yaml import load
from yaml import CLoader as Loader
#plotting:
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#abil functions:
from abil.tune import tune
from abil.predict import predict
from abil.post import post
from abil.utils import example_data 
#paths:
import os

os.chdir(os.path.join(".", "docs", "examples"))

#load configuration yaml:
with open('regressor.yml', 'r') as f:
    model_config = load(f, Loader=Loader)
#load example training data:
d = pd.read_csv(os.path.join("..", "data", "training.csv"))
#define target:
target = "Gephyrocapsa huxleyi HET"
#define predictors based on YAML:
predictors = model_config['predictors']
#split training data in X_train and y
y = d[target]
X_train = d[predictors]

#train your model:
m = tune(X_train, y, model_config)
m.train(model="rf")
m.train(model="xgb")
m.train(model="knn")

#load prediction data:
X_predict = pd.read_csv(os.path.join("..", "data", "southern_ocean_surface.csv"))
X_predict.set_index(['lat', 'lon'], inplace=True)

#predict your model:
m = predict(X_train, y, X_predict, model_config)
m.make_prediction()

# Posts
targets = np.array([target])
def do_post(statistic):
    m = post(X_train, y, X_predict, model_config, statistic, datatype="poc")
    m.estimate_carbon("pg poc")
    m.export_ds("my_first_regressor_model")

do_post(statistic="mean")
do_post(statistic="ci95_UL")
do_post(statistic="ci95_LL")

# Load the predictions
ds = xr.open_dataset("./ModelOutput/regressor/posts/my_first_regressor_model_mean_poc.nc")
ds_UL = xr.open_dataset("./ModelOutput/regressor/posts/my_first_regressor_model_ci95_UL_poc.nc")
ds_LL = xr.open_dataset("./ModelOutput/regressor/posts/my_first_regressor_model_ci95_LL_poc.nc")

# Log-transform data
ds['Gephyrocapsa huxleyi HET'] = np.log10(ds['Gephyrocapsa huxleyi HET'] + 1)
ds_UL['Gephyrocapsa huxleyi HET'] = np.log10(ds_UL['Gephyrocapsa huxleyi HET'] + 1)
ds_LL['Gephyrocapsa huxleyi HET'] = np.log10(ds_LL['Gephyrocapsa huxleyi HET'] + 1)
d['Gephyrocapsa huxleyi HET'] = np.log10(d['Gephyrocapsa huxleyi HET'] + 1)

# Create the figure
fig, axs = plt.subplots(
    2, 2, figsize=(8, 6),
    subplot_kw={'projection': ccrs.SouthPolarStereo()}
)
# Flatten for indexing
ax00, ax01, ax10, ax11 = axs.flat
for ax in (ax00, ax01, ax10, ax11):
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())  # Southern Ocean
    ax.coastlines()
    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, color='gray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

#define titles and labels
titles = ['Training Data', 'Mean Abundance', '95% CI Lower Limit', '95% CI Upper Limit']
panel_labels = ['A)', 'B)', 'C)', 'D)']

def add_title(ax, title, label, y=1.05):
    ax.set_title(f'$\mathbf{{{label}}}$  {title}', loc='left', pad=10, y=y, fontsize=10)

#subplot A - training data
sc = ax00.scatter(
    d['lon'], d['lat'],
    c=d['Gephyrocapsa huxleyi HET'],
    cmap='viridis', s=10,
    transform=ccrs.PlateCarree(),
    vmin=0
)
add_title(ax00, titles[0], panel_labels[0])
cbar0 = plt.colorbar(sc, ax=ax00, shrink=0.6, pad=0.1)
cbar0.ax.tick_params(labelsize=8)
cbar0.set_label('log$_{10}$ abundance (cells L$^{-1}$)', size=8)

#subplot B - mean POC
p1 = ds['Gephyrocapsa huxleyi HET'].plot(
    ax=ax01, cmap='viridis', add_colorbar=False, 
    transform=ccrs.PlateCarree()
)
add_title(ax01, titles[1], panel_labels[1])
cbar1 = plt.colorbar(p1, ax=ax01, shrink=0.6, pad=0.1)
cbar1.ax.tick_params(labelsize=8)
cbar1.set_label('log$_{10}$ abundance (cells L$^{-1}$)', size=8)

#subplot C - 95% CI Lower POC
p2 = ds_LL['Gephyrocapsa huxleyi HET'].plot(
    ax=ax10, cmap='viridis', add_colorbar=False, 
    transform=ccrs.PlateCarree()
)
add_title(ax10, titles[2], panel_labels[2])
cbar2 = plt.colorbar(p2, ax=ax10, shrink=0.6, pad=0.1)
cbar2.ax.tick_params(labelsize=8)
cbar2.set_label('log$_{10}$ abundance (cells L$^{-1}$)', size=8)

#subplot D - 95% CI Upper POC
p3 = ds_UL['Gephyrocapsa huxleyi HET'].plot(
    ax=ax11, cmap='viridis', add_colorbar=False, 
    transform=ccrs.PlateCarree()
)
add_title(ax11, titles[3], panel_labels[3])
cbar3 = plt.colorbar(p3, ax=ax11, shrink=0.6, pad=0.1)
cbar3.ax.tick_params(labelsize=8)
cbar3.set_label('log$_{10}$ abundance (cells L$^{-1}$)', size=8)

plt.tight_layout()
plt.savefig('figure_1.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
