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
from matplotlib.colors import Normalize
#abil functions:
from abil.tune import tune
from abil.predict import predict
from abil.post import post
from abil.utils import example_data 
#paths:
import os

os.chdir(os.path.join(".", "docs", "examples"))

#load configuration yaml:
with open('2-phase.yml', 'r') as f:
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
X_predict = pd.read_csv(os.path.join("..", "data", "southern_averaged.csv"))
X_predict.set_index(['lat', 'lon'], inplace=True)

#predict your model:
m = predict(X_train, y, X_predict, model_config)
m.make_prediction()

# Posts
targets = np.array([target])
def do_post(statistic):
    m = post(X_train, y, X_predict, model_config, statistic, datatype="poc")
    m.estimate_carbon("pg poc")
    m.export_ds("my_first_2-phase_model")

do_post(statistic="mean")
do_post(statistic="ci95_UL")
do_post(statistic="ci95_LL")

# Load the predictions
ds = xr.open_dataset("./ModelOutput/2-phase/posts/my_first_2-phase_model_mean_poc.nc")
ds_UL = xr.open_dataset("./ModelOutput/2-phase/posts/my_first_2-phase_model_ci95_UL_poc.nc")
ds_LL = xr.open_dataset("./ModelOutput/2-phase/posts/my_first_2-phase_model_ci95_LL_poc.nc")

# Create the figure
def plot_panel(ax, data, var, title, label, cmap='viridis', cbar_label='abundance (cells L$^{-1}$)'):
    ax.set_extent([-180,180,-90,-30], crs=ccrs.PlateCarree()); ax.coastlines(); ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, color='gray'); ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_title(f'$\mathbf{{{label}}}$  {title}', loc='left', pad=10, y=1.05, fontsize=10)

    if isinstance(data, pd.DataFrame):  # raw points -> hexbin (no xarray conversion)
        lon, lat, vals = np.asarray(data['lon']), np.asarray(data['lat']), np.asarray(data[var])
        vmin, vmax = np.nanpercentile(vals, [2,98]); x, y = ax.projection.transform_points(ccrs.PlateCarree(), lon, lat)[:,0:2].T
        h = ax.hexbin(x, y, C=vals, reduce_C_function=np.nanmedian, gridsize=20, mincnt=1, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    else:  # gridded data -> xarray plot
        da = data[var] if isinstance(data, xr.Dataset) else data
        h = da.plot(ax=ax, cmap=cmap, add_colorbar=False, robust=True, transform=ccrs.PlateCarree())

    cb = plt.colorbar(h, ax=ax, shrink=0.6, pad=0.1); cb.ax.tick_params(labelsize=8); cb.set_label(cbar_label, size=8)

# ---- usage ----
fig, axs = plt.subplots(2,2, figsize=(8,6), subplot_kw={'projection': ccrs.SouthPolarStereo()})
(ax00, ax01), (ax10, ax11) = axs
v = 'Gephyrocapsa huxleyi HET'
plot_panel(ax00, d,     v, 'Training Data',          'A)')
plot_panel(ax01, ds,    v, 'Mean Abundance',         'B)')
plot_panel(ax10, ds_LL, v, '95% CI Lower Limit',     'C)')
plot_panel(ax11, ds_UL, v, '95% CI Upper Limit',     'D)')
plt.tight_layout()
plt.savefig('figure_2-phase.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
