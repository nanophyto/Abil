"""
2-phase regressor
"""
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from yaml import load, CLoader as Loader
from abil.tune import tune
from abil.predict import predict
from abil.post import post
from abil.utils import upsample

# Load YAML config
with open('2-phase.yml', 'r') as f:
    model_config = load(f, Loader=Loader)

# Define targets and features:
targets = ['Emiliania huxleyi', 'Florisphaera profunda', 'Umbellosphaera irregularis']
features = ['temperature', 'PAR', 'no3']
# Observations
d = pd.read_csv("./data/training.csv")
d.set_index(['depth', 'time'], inplace=True)
# Environmental data:
X_predict =  pd.read_csv("./data/prediction.csv")
X_predict.set_index(['depth', 'time'], inplace=True)


def plotting(d, targets):
    try:
        ds = d.to_xarray()
    except:
        ds = d
    # Create a figure with 1 row and 3 columns for subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  
    # Plot each variable in its respective subplot
    ds[targets[0]].plot(ax=axes[0], yincrease=False)
    axes[0].set_title(targets[0])
    ds[targets[1]].plot(ax=axes[1], yincrease=False)
    axes[1].set_title(targets[1])
    ds[targets[2]].plot(ax=axes[2], yincrease=False)
    axes[2].set_title(targets[2])

plotting(d, targets)
plt.show()
plt.savefig('observational_data.png')
plt.close()

plotting(X_predict, features)
plt.show()
plt.savefig('environmental_data.png')
plt.close()

def predict_single_species(df, target):
    # Pseudo-absences:
    df[target] = df[target].fillna(0)
    df = upsample(df, target, ratio=10)
    # Define y and X: 
    y = df[target]
    X_train = df[features]
    # Train models
    m = tune(X_train, y, model_config)
    m.train(model="rf", regressor=True)
    m.train(model="xgb", regressor=True)
    m.train(model="knn", regressor=True)
    # Predict
    m = predict(X_train, y, X_predict, model_config)
    m.make_prediction()

# Apply model optimization and prediction to each species:
# predict_single_species(d, 'Emiliania huxleyi')
# predict_single_species(d, 'Florisphaera profunda')
# predict_single_species(d, 'Umbellosphaera irregularis')

# Post-process
y = d[targets]
X_train = d[features]
# m = post(X_train, y, X_predict, model_config, "mean")
# m.export_ds("2-phase_model")

#Final plotting
ds = xr.open_dataset("ModelOutput/2-phase/posts/2-phase_model_mean.nc")
plotting(ds, targets)
plt.show()
plt.savefig('species_predictions.png')
plt.close()