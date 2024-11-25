#%%
# Merge monthly WOA18 Temperature and subset to 0-200m

import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd

base_path = '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/raw_data/WOA18/'
files = ['01',
         '02',
         '03',
         '04',
         '05',
         '06',
         '07',
         '08',
         '09',
         '10',
         '11',
         '12']
ds_all = []
for i in range(0, 12):
    ds = xr.open_dataset(base_path + "woa18_decav_t" + files[i] + "_01.nc",
                         decode_times=False,drop_variables={'climatology_bounds','crs','depth_bnds',
                                                            't_dd','t_gp','t_ma','t_mn','t_oa','t_sd','t_se',
                                                            'lat_bnds','lon_bnds','time'}
                         )
    df = ds.to_dataframe()
    df.reset_index(inplace=True)
    df = df[df["depth"] <= 200]
    print(df)
    df = df.set_index(['depth','lat','lon'])
    ds = df.to_xarray()
    ds = ds.interp(depth=np.arange(0,205,5))
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1)),
                    'lon': (['lon'], np.arange(-180, 180, 1)),
                    'depth': (['depth'], np.arange(0, 205, 5))
                    })
    regridder1 = xe.Regridder(ds, ds_out, method="bilinear", periodic=True)
    dr_out = regridder1(ds['t_an'])
    df = dr_out.to_dataframe(name="temperature")
    df.reset_index(inplace = True)
    df['time'] = i+1
    df = df.groupby(['time', 'depth', 'lat', 'lon']).mean().reset_index()
    df.set_index(['time', 'depth', 'lat', 'lon'], inplace=True)
    print(df)
    ds = df.to_xarray()
    ds_all.append(ds)

ds = xr.merge(ds_all)
df = ds.to_dataframe()
df.reset_index(inplace = True)
df.set_index(['time', 'depth', 'lat', 'lon'], inplace=True)
ds = df.to_xarray()

ds.to_netcdf("/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/temperature.nc")

print("fin")

# %%
