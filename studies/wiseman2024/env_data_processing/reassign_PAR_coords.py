import numpy as np
import xarray as xr

# Load the NetCDF file
file_path = '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/raw_data/RS_PAR/RS_PAR_ESM-based_fill_monthly_clim_1998-2022.nc'
ds = xr.open_dataset(file_path, mode="r+")

# Rename coordinates
ds = ds.rename({'latitude':'lat','longitude':'lon','Depth':'depth','Date':'time','Time':'time'})

# Save the changes
ds.to_netcdf('/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/par.nc')

# Close the dataset
ds.close()