# %%

import numpy as np
import xarray as xr
import xesmf as xe

# Define the file list
files = [
    '20030101_20230131',
    '20030201_20230228',
    '20030301_20220331',
    '20030401_20210430',
    '20030501_20220531',
    '20030601_20220630',
    '20020701_20230731',
    '20020801_20230831',
    '20020901_20230930',
    '20021001_20211031',
    '20021101_20211130',
    '20021201_20221231'
]

base_path = '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/raw_data/MODIS_Rrs547/'

# Create a target grid for regridding
ds_out = xr.Dataset({
    'lat': (['lat'], np.arange(-90, 90, 1)),
    'lon': (['lon'], np.arange(-180, 180, 1))
})

# Create a filename for storing the weights
weights_file = "regridded_data/regrid_weights.nc"

print('Initializing regridder')
# Open the first dataset to create a persistent regridder (only calculated once)
initial_ds = xr.open_dataset(f"{base_path}AQUA_MODIS.{files[0]}.L3m.MC.RRS.Rrs_547.9km.nc")
regridder = xe.Regridder(initial_ds, ds_out, method="bilinear", periodic=True, filename=weights_file)

ds_all = []

for i in range(0, 12):
    ds = xr.open_dataset(f"{base_path}AQUA_MODIS.{files[i]}.L3m.MC.RRS.Rrs_547.9km.nc")
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1)),
                    'lon': (['lon'], np.arange(-180, 180, 1))
                    })
    print(ds)
    # Regrid chlor_a to the new lat/lon grid (with persistent regridder, reuse weights)
    regridder = xe.Regridder(ds, ds_out, method="bilinear", periodic=True, reuse_weights=True, filename=weights_file)
    dr_out = regridder(ds['Rrs_547'])
    
    # Add time and depth dimensions in one step
    dr_out = dr_out.assign_coords(time=i+1).expand_dims(time=[i+1])
    dr_out_depth = dr_out.expand_dims(depth=[0, 205])
    
    # Interpolate along the depth axis from 0 to 305 meters at 5m intervals
    dr_out_depth = dr_out_depth.interp(depth=np.arange(0, 205, 5))
    
    # Append the dataset to the list
    ds_all.append(dr_out_depth)

# Concatenate all datasets along the time dimension
ds = xr.concat(ds_all, dim="time")

# Additionally, fill NaN regions (high latitudes during winter) with min or 10% of maximum, whichever is lower
# If you want to skip this, skip lines 64-89 and swtich "filled_ds" to "ds" for lines 90-96
# Calculate the max and min along the time dimension
max_values = ds.max(dim='time', skipna=True)
print(max_values)
min_values = ds.min(dim='time', skipna=True)
print(min_values)

# Print the sizes of max_values and min_values
print("Size of max_values:", max_values.sizes)
print("Size of min_values:", min_values.sizes)

# Calculate 10% of the max values
ten_percent_max = 0.1 * max_values

# Create a mask for locations with missing data (more than 0 and less than 12 NaNs)
null_counts = ds.isnull().sum(dim='time')
mask = (null_counts > 0) & (null_counts < 12)

# Create an array to hold the fill values
fill_values = xr.where(mask, np.minimum(ten_percent_max, min_values), np.nan)

# Fill missing values only where they are present in the original dataset
filled_ds = ds.where(ds.notnull(), fill_values)

# Ensure not to fill if all values are missing (i.e., keep original data if all months are NaN)
filled_ds = xr.where(null_counts == 12, ds, filled_ds)

# Assign a name to the DataArray
filled_ds.name = 'Rrs_547'

# Save the filled dataset to NetCDF
filled_ds.to_netcdf("/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/Rrs_547.nc")

print("Finished processing and saved to Rrs_547.nc")
# %%