# %%
from operator import index
import numpy as np
import xarray as xr
import xesmf as xe

path = "/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/raw_data/RS_PAR/RS_PAR_ESM-based_fill_monthly_clim_1998-2022_0-200.nc"
name = "PAR"

#load netcdf as xarray
ds1 = xr.open_dataset(path)
ds1 = ds1.drop_vars(['DOI'])
ds1 = ds1.rename(name_dict = {'latitude':'lat', 'longitude':'lon', 'Time':'time', 'Depth':'depth'})

# setup new grid to regrid to:
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1)),
                    'lon': (['lon'], np.arange(-180, 180, 1)),
                    'depth': (['depth'], np.arange(0, 200, 5)),
                    'time': (['time'], np.arange(1, 13, 1)),
                    })

regridder1 = xe.Regridder(ds1, ds_out, 'conservative', periodic=True)
dr1_out = regridder1(ds1['PAR'],skipna=True, na_thres=0.75)

df1 = dr1_out.to_dataframe(name=name).reset_index()
df1.set_index(['time', 'depth', 'lat', 'lon'], inplace=True)
print(df1)
ds = df1.to_xarray()
print('saving ' + name)
ds.to_netcdf("/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/" + name + ".nc")

print("fin")
# %%
