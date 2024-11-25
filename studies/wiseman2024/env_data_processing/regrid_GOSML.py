# %%
# Regrid GOSML Mixed Layer Depth to 180x360x12 and repeat for depths 0-200m
from operator import index
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
path = "/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/raw_data/GOSML/mixed_layer_properties_mean.nc"


ds = xr.open_dataset(path)
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1)),
                     'lon': (['lon'], np.arange(-180, 180, 1))
                     })
print(ds)
regridder1 = xe.Regridder(ds, ds_out, method="bilinear", periodic=True)
dr_out = regridder1(ds['depth_mean'])
df = dr_out.to_dataframe(name="mld")
df.reset_index(inplace = True)
df['month'] = df['month'] + 0.5
df1 = df 
df2 = df
df1 = df1.assign(depth=0)
df2 = df2.assign(depth=205)
df = pd.concat([df1, df2])
df.rename(columns = {'latitude':'lat', 'longitude':'lon', 'month':'time'}, inplace = True)
df = df.groupby(['time', 'depth', 'lat', 'lon']).mean().reset_index()
df.set_index(['time', 'depth', 'lat', 'lon'], inplace=True)
print(df)
ds = df.to_xarray()
ds = ds.interp(depth=np.arange(0, 205, 5))
df = ds.to_dataframe()
df.reset_index(inplace = True)
df.set_index(['time', 'depth', 'lat', 'lon'], inplace=True)
ds = df.to_xarray()
ds.to_netcdf("/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/MLD.nc")

print("fin")
# %%
