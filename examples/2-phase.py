"""
2-phase Ensemble 
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
from abil.utils import example_data 
#plotting:
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

os.chdir(os.path.join(".", "examples"))

#load configuration yaml:
with open('2-phase.yml', 'r') as f:
    model_config = load(f, Loader=Loader)

#load example training data:
d = pd.read_csv(os.path.join("data", "so_training.csv"))
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
X_predict = pd.read_csv(os.path.join("data", "so_prediction.csv"))
X_predict.set_index(['lat', 'lon'], inplace=True)
X_predict = X_predict[predictors]

#predict your model:
m = predict(X_train, y, X_predict, model_config)
m.make_prediction()

# Posts
targets = np.array([target])
def do_post(statistic):
    m = post(X_train, y, X_predict, model_config, statistic, datatype="abundance")
    m.export_ds("SO") #Southern Ocean

do_post(statistic="mean")
do_post(statistic="ci95_UL")
do_post(statistic="ci95_LL")

# Load the predictions
ds = xr.open_dataset(os.path.join("ModelOutput", "2-phase", "posts", "SO_mean_abundance.nc"))
ds_LL = xr.open_dataset(os.path.join("ModelOutput", "2-phase", "posts", "SO_ci95_LL_abundance.nc"))
ds_UL = xr.open_dataset(os.path.join("ModelOutput", "2-phase", "posts", "SO_ci95_UL_abundance.nc"))

# Create the figure
def plot_panel(ax, data, var, title, label, 
               cbar_label='abundance (cells L$^{-1}$)'):
    ax.set_extent([-180,180,-90,-30], crs=ccrs.PlateCarree())
    ax.coastlines(); ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, color='gray') 
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_title(f'$\mathbf{{{label}}}$  {title}', loc='left', 
                 pad=10, y=1.05, fontsize=10)

    if isinstance(data, pd.DataFrame):  # raw points
        lon, lat = np.asarray(data['lon']), np.asarray(data['lat'])
        vals = np.asarray(data[var])
        # clip outliers for plotting
        vmin, vmax = np.nanpercentile(vals, [2,98])
        x, y = ax.projection.transform_points(ccrs.PlateCarree(), lon, lat)[:,0:2].T
        h = ax.hexbin(x, y, C=vals, reduce_C_function=np.nanmean, gridsize=20, 
                      mincnt=1, norm=Normalize(vmin=vmin, vmax=vmax))
    else:  # gridded data
        da = data[var] if isinstance(data, xr.Dataset) else data
        h = da.plot(ax=ax, add_colorbar=False, robust=True, 
                    transform=ccrs.PlateCarree())

    cb = plt.colorbar(h, ax=ax, shrink=0.6, pad=0.1); cb.ax.tick_params(labelsize=8)
    cb.set_label(cbar_label, size=8)

# apply plotting function and export
fig, axs = plt.subplots(2,2, figsize=(8,6), 
                        subplot_kw={'projection': ccrs.SouthPolarStereo()})
(ax00, ax01), (ax10, ax11) = axs
plot_panel(ax00, d, target, 'Training Data', 'A)')
plot_panel(ax01, ds,    target, 'Mean Abundance', 'B)')
plot_panel(ax10, ds_LL, target, '95% CI Lower Limit', 'C)')
plot_panel(ax11, ds_UL, target, '95% CI Upper Limit', 'D)')
plt.tight_layout()
plt.savefig('SO_predictions.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# we can also look at integrated totals (stocks)
def do_integrations(statistic):
    m = post(X_train, y, X_predict, model_config, statistic, datatype="pic")
    m.estimate_carbon("pg pic")
    vol_conversion = 1e3 # to convert from pg C L-1 to pg C m-3
    magnitude_conversion = 1e-24 # convert from pg C to Tg C
    vol_conversion = vol_conversion*200 # model approximates the top 200m
    integ = m.integration(m, vol_conversion=vol_conversion,
                          magnitude_conversion=magnitude_conversion)
    integ.integrated_totals(targets)

do_integrations(statistic="mean")
do_integrations(statistic="ci95_UL")
do_integrations(statistic="ci95_LL")

mean = pd.read_csv(os.path.join("ModelOutput", "2-phase", "posts", 
                                "integrated_totals", 
                                "ens_integrated_totals_mean_pic.csv"))['total'][0]
ci95_LL = pd.read_csv(os.path.join("ModelOutput", "2-phase", "posts", 
                                "integrated_totals", 
                                "ens_integrated_totals_ci95_LL_pic.csv"))['total'][0]
ci95_UL = pd.read_csv(os.path.join("ModelOutput", "2-phase", "posts", 
                                "integrated_totals", 
                                "ens_integrated_totals_ci95_UL_pic.csv"))['total'][0]

print(f"estimated integrated total: {mean:.2f} [{ci95_LL:.2f}, {ci95_UL:.2f}] Tg IC")
