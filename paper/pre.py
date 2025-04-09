import xarray as xr 
import pandas as pd

ds = xr.open_dataset('/home/phyto-2/Abil_SDM_data/env_data.nc')

ds = ds[["temperature", 
    "sio4", "po4", "no3", 
    "o2", "DIC", "TA",
    "PAR"]]

ds = ds.isel(depth=1).mean(dim=["time"]) 

d = ds.to_dataframe()

d.to_csv("./env_mean_global_surface.csv")