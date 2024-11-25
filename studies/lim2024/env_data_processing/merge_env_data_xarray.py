#%%
import xarray as xr
import numpy as np

# List of dataset file paths, to add more fields, just add path to .nc file
# Data must already be on a common grid (example data is 180x360x41x12)
file_paths = [
    '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/temperature.nc',
    '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/sio4.nc',
    '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/po4.nc',
    '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/no3.nc',
    '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/o2.nc',
    '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/MLD.nc',
    '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/DIC.nc',
    '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/TA.nc',
    '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/par.nc',
    '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/chlor_a.nc',
    '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/Rrs_547.nc',
    '/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/Rrs_667.nc'    
]

# Open all datasets
datasets = [xr.open_dataset(fp) for fp in file_paths]

# Align all datasets to the intersection of their coordinates
aligned_datasets = xr.align(*datasets, join='inner')

# Merge the aligned datasets
merged_ds = xr.merge(aligned_datasets)

# Drop DOI
merged_ds = merged_ds.drop_vars(['DOI'])

# Calculate CI_2
merged_ds['CI_2'] = merged_ds['Rrs_547'] - merged_ds['Rrs_667']

# Drop Rrs_547 and Rrs_667
merged_ds = merged_ds.drop_vars(['Rrs_547','Rrs_667'])

# List of variables of interest
variables_of_interest = ['temperature','sio4', 'po4', 'no3','o2','mld','DIC','TA','PAR','chlor_a','CI_2']  # Add all relevant variable names

# Create a mask for where any of the variables are NaN
mask = xr.concat([merged_ds[var].isnull() for var in variables_of_interest], dim='var').any(dim='var')

# Apply the mask to all variables, setting values to NaN where any variable is missing
for var in variables_of_interest:
    merged_ds[var] = merged_ds[var].where(~mask, np.nan)

# Convert the xarray Dataset to a Pandas DataFrame
df = merged_ds.to_dataframe()

# Reset index to flatten the multi-index DataFrame and drop NA
df = df.reset_index()
df.dropna(inplace=True)

# Save the DataFrame to a CSV file
df.to_csv('/home/mv23682/Documents/Abil/studies/wiseman2024/data/env_data.csv', index=False)

merged_ds['lat'].attrs['units'] = 'degrees_north'
merged_ds['lat'].attrs['long_name'] = 'latitude'

merged_ds['lon'].attrs['units'] = 'degrees_east'
merged_ds['lon'].attrs['long_name'] = 'longitude'

merged_ds['depth'].attrs['units'] = 'm'
merged_ds['depth'].attrs['positive'] = 'down'

merged_ds['time'].attrs['units'] = 'months'

merged_ds['chlor_a'].attrs['units'] = 'mg.m-3'
merged_ds['chlor_a'].attrs['long_name'] = 'chlorophyll_a'
merged_ds['chlor_a'].attrs['description'] = 'Aqua MODIS Level 3 binned chlorophyll data monthly climatology, version R2022.0, 2002-07-01 to 2023-09-03 (NASA Ocean Biology Processing Group, 2022)'

merged_ds['DIC'].attrs['units'] = 'umol.kg-1'
merged_ds['DIC'].attrs['long_name'] = 'dissolved inorganic carbon'
merged_ds['DIC'].attrs['description'] = 'Monthly climatology of total dissolved inorganic carbon (TCO2) centered in 1995 and obtained with NNGv2LDEO of Broullón et al. (2020)'

merged_ds['no3'].attrs['units'] = 'umol.kg-1'
merged_ds['no3'].attrs['long_name'] = 'nitrate'
merged_ds['no3'].attrs['description'] = 'Objectively analyzed mean fields for moles_concentration_of_nitrate_in_sea_water from WOA18 of Garcia et al. (2019)'

merged_ds['mld'].attrs['units'] = 'm'
merged_ds['mld'].attrs['long_name'] = 'mapped mean mixed layer depth'
merged_ds['mld'].attrs['description'] = 'Global Ocean Surface Mixed Layer Statistical Monthly Climatology (GOSML) mean mixed layer depth from Johnson & Lyman (2022)'

merged_ds['o2'].attrs['units'] = 'umol.kg-1'
merged_ds['o2'].attrs['long_name'] = 'dissolved oxygen'
merged_ds['o2'].attrs['description']= 'Objectively analyzed mean fields for mole_concentration_of_dissolved_molecular_oxygen_in_sea_water from WOA18 of Garcia et al. (2019)'

merged_ds['PAR'].attrs['units'] = 'W.m-2'
merged_ds['PAR'].attrs['long_name'] = 'photosynthetically activate radiation'
merged_ds['PAR'].attrs['description'] = 'RS_PAR_ESM-based_fill_monthly_clim_1998-2022 from Castant et al. (2024)'

merged_ds['po4'].attrs['units'] = 'umol.kg-1'
merged_ds['po4'].attrs['long_name'] = 'phosphate'
merged_ds['po4'].attrs['description'] = 'Objectively analyzed mean fields for moles_concentration_of_phosphate_in_sea_water from WOA18 of Garcia et al. (2019)'

merged_ds['sio4'].attrs['units'] = 'umol.kg-1'
merged_ds['sio4'].attrs['long_name'] = 'silicate'
merged_ds['sio4'].attrs['description'] = 'Objectively analyzed mean fields for moles_concentration_of_silicate_in_sea_water from WOA18 of Garcia et al. (2019)'

merged_ds['TA'].attrs['units'] = 'umol.kg-1'
merged_ds['TA'].attrs['long_name'] = 'total alkalinity'
merged_ds['TA'].attrs['description'] = 'Monthly climatology of total alkalinity obtained with NNGv2 of Broullón et al. (2019)'

merged_ds['temperature'].attrs['units'] = 'degrees_celsius'
merged_ds['temperature'].attrs['long_name'] = 'sea_water_temperature'
merged_ds['temperature'].attrs['description'] = 'Objectively analyzed mean fields for sea_water_temperature from WOA18 of Locarnini et al. (2019)'

merged_ds['CI_2'].attrs['units'] = 'sr-1'
merged_ds['CI_2'].attrs['long_name'] = 'Color Index 2'
merged_ds['CI_2'].attrs['description'] = 'Color Index 2 (CI2) from Mitchell et al. (2017) calculated using Aqua MODIS Level 3 binned Rrs_547 and Rrs_667 data monthly climatology, version R2022.0, 2002-07-01 to 2023-09-03 (NASA Ocean Biology Processing Group, 2022)'

# Save the result to a new NetCDF file
merged_ds.to_netcdf('/home/mv23682/Documents/Abil/studies/wiseman2024/data/env_data.nc')
print('fin')
# %%
