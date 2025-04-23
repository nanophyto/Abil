#%%
# Merge monthly WOA18 Oxygen and subset to 0-200m

import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd

base_path = '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/raw_data/WOA18/'
files = [f"{i:02}" for i in range(1, 13)]

sample_ds = xr.open_dataset(base_path + "woa18_all_o01_01.nc", decode_times=False,
                            drop_variables={'climatology_bounds','crs','depth_bnds',
                                            'o_dd','o_gp','o_ma','o_mn','o_oa','o_sd','o_se',
                                            'lat_bnds','lon_bnds','time'})

sample_df = sample_ds.to_dataframe().reset_index()
sample_df = sample_df[sample_df["depth"] <= 200]
sample_df = sample_df.set_index(['depth', 'lat', 'lon'])
sample_xr = sample_df.to_xarray().interp(depth=np.arange(0, 205, 5))
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1)),
                    'lon': (['lon'], np.arange(-180, 180, 1)),
                    'depth': (['depth'], np.arange(0, 205, 5))
                    })
regridder1 = xe.Regridder(sample_xr, ds_out, method="conservative", periodic=True)

ds_all = []
for i in range(0, 12):
    print(f"Processing month {i+1}...")
    ds = xr.open_dataset(base_path + "woa18_all_o" + files[i] + "_01.nc",
                         decode_times=False,drop_variables={'climatology_bounds','crs','depth_bnds',
                                                            'o_dd','o_gp','o_ma','o_mn','o_oa','o_sd','o_se',
                                                            'lat_bnds','lon_bnds','time'}
                         )
    df = ds.to_dataframe().reset_index()
    df = df[df["depth"] <= 200]
    df = df.set_index(['depth', 'lat', 'lon'])
    ds = df.to_xarray().interp(depth=np.arange(0, 205, 5))
    dr_out = regridder1(ds['o_an'],skipna=True, na_thres=0.75)
    df = dr_out.to_dataframe(name="o2")
    df.reset_index(inplace = True)
    df['time'] = i+1
    df = df.groupby(['time', 'depth', 'lat', 'lon']).mean().reset_index()
    df.set_index(['time', 'depth', 'lat', 'lon'], inplace=True)
    ds = df.to_xarray()
    ds_all.append(ds)

ds = xr.merge(ds_all)
print("Regridding complete.")
df = ds.to_dataframe().reset_index()
df.set_index(['time', 'depth', 'lat', 'lon'], inplace=True)
ds = df.to_xarray().to_netcdf("/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/o2.nc")

print("fin")

# %%
