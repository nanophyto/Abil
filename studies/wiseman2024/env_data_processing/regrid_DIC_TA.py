# %%
from operator import index
import numpy as np
import xarray as xr
import xesmf as xe

def preprocess(path, name):
  #load netcdf as xarray
  ds1 = xr.open_dataset(path)
  df1 = ds1.to_dataframe()
  df1.reset_index(inplace = True)
  df1 =  df1[df1["depth"] <= 200]
  print(df1)
  df1.rename(columns = {'latitude':'lat', 'longitude':'lon'}, inplace = True)
  df1.set_index(['time', 'depth', 'lat', 'lon'], inplace=True)
  ds1 = df1.to_xarray()

  ds1 = ds1.interp(depth=np.arange(0, 205, 5))


  # setup new grid to regrid to:
  ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1)),
                        'lon': (['lon'], np.arange(-180, 180, 1)),
                        'depth': (['depth'], np.arange(0, 200, 5)),
                        'time': (['time'], np.arange(1, 13, 1)),
                      })

  regridder1 = xe.Regridder(ds1, ds_out, 'bilinear', periodic=True)
  if (name=="DIC"):
    dr1_out = regridder1(ds1['TCO2_NNGv2LDEO'])
  elif (name=="TA"):
      dr1_out = regridder1(ds1['AT_NNGv2'])


  df1 = dr1_out.to_dataframe(name=name)
  df1.reset_index(inplace = True)
  if (name=="DIC"):
    df1['DIC'].mask(df1['DIC'] == 0, inplace=True)
  elif (name=="TA"):
    df1['TA'].mask(df1['TA'] == 0, inplace=True)
  df1.set_index(['time', 'depth', 'lat', 'lon'], inplace=True)
  print(df1)
  ds = df1.to_xarray()
  print('saving ' + name)
  ds.to_netcdf("/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/" + name + ".nc")

# Run for both DIC and TA
preprocess("/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/raw_data/NNGv2/TCO2_NNGv2LDEO_climatology.nc", "DIC")
preprocess("/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/raw_data/NNGv2/AT_NNGv2_climatology.nc", "TA")

print("fin")
# %%
