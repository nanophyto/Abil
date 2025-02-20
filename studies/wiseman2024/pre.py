#%%
import pandas as pd
import numpy as np
import xarray as xr

# Load raw dataset from Marsh et al.
d_raw = pd.read_csv('/home/mv23682/Documents/Abil/studies/wiseman2024/data/calcif_2018_v2.0.1.csv',
                 skiprows=1,
                 names=["PI","Expedition","OS Region","Reference_Author_Published_year","Reference_doi",
                        "Date","Sample_ID","Latitude","Longitude","Depth","Irr_Depth",
                        "Optical_Depth","Method","Incubation_Length",
                        "Calcification","Calcification_Standard Deviation",
                        "Primary_Production","Primary_Production_Standard_Deviation",
                        "0.2-2 um Net Primary Production [µmol C m-3 d-1]","0.2-2 um Net Primary Production_Standard Deviation  [µmol C m-3 d-1]",
                        "2-10 um Net Primary Production  [µmol C m-3 d-1]","2-10 um Net Primary Production_Standard Deviation [µmol C m-3 d-1]",
                        ">10 um Net Primary Production  [µmol C m-3 d-1]",">10 um Net Primary Production_Standard Deviation [µmol C m-3 d-1]",
                        "Total Coccolithophore cell counts [cells mL-1]","Emiliania huxleyi cell counts [cells mL-1]",
                        "Chlorophyll-a [mg m-3)","NOx (µM/L)","Silicate (µM/L)","Phosphate (µM/L)","DIC [μmol/kg-seawater]",
                        "Total Alkalinity  [μmol/kg-seawater]","Bicarbonate (HCO3) [μmol/kg-seawater]",
                        "Carbonate (CO3) [μmol/kg-seawater]","pH","Temperature (degrees C)","Salinity (ppt)"
                        ])

d_raw['DateTime'] = pd.to_datetime(d_raw['Date'],dayfirst=True)
d_raw['Month'] = pd.DatetimeIndex(d_raw['DateTime']).month
d_raw['Year'] = pd.DatetimeIndex(d_raw['DateTime']).year
# Drop rows where method is Diff or Ca45 due to quality control concerns (only applicable for CP Data)
d_filtered = d_raw[~d_raw['Method'].isin(['Diff','Ca45'])]
print(d_filtered["Calcification"].notna().sum())

# Drop rows where the condition "Calcification / Emiliania_huxleyi_cell_counts > 3.5" is met (only applicable for CP Data)
# Only apply condition where "Emiliania huxleyi cell counts [cells mL-1]" is not NaN
# This removes extraneously high data values that may have quality control concerns
mask = d_filtered["Emiliania huxleyi cell counts [cells mL-1]"].notna() & (
    d_filtered["Calcification"] / d_filtered["Emiliania huxleyi cell counts [cells mL-1]"] > 3.5)
d_filtered = d_filtered[~mask]
print(d_filtered["Calcification"].notna().sum())
d_filtered = d_filtered[d_filtered["Calcification"] <= 1000]
d = d_filtered
print(d["Calcification"].notna().sum())

# Drop data unnecessary for Abil.py
d = d.convert_dtypes()
d = d.drop(["PI","Expedition","OS Region","Reference_Author_Published_year","Reference_doi",
                        "Sample_ID","Irr_Depth",
                        "Optical_Depth","Method","Incubation_Length",
                        "Primary_Production",
                        "Primary_Production_Standard_Deviation",
                        "0.2-2 um Net Primary Production [µmol C m-3 d-1]","0.2-2 um Net Primary Production_Standard Deviation  [µmol C m-3 d-1]",
                        "2-10 um Net Primary Production  [µmol C m-3 d-1]","2-10 um Net Primary Production_Standard Deviation [µmol C m-3 d-1]",
                        ">10 um Net Primary Production  [µmol C m-3 d-1]",">10 um Net Primary Production_Standard Deviation [µmol C m-3 d-1]",
                        "Total Coccolithophore cell counts [cells mL-1]","Emiliania huxleyi cell counts [cells mL-1]",
                        "Chlorophyll-a [mg m-3)","NOx (µM/L)","Silicate (µM/L)","Phosphate (µM/L)","DIC [μmol/kg-seawater]",
                        "Total Alkalinity  [μmol/kg-seawater]","Bicarbonate (HCO3) [μmol/kg-seawater]",
                        "Carbonate (CO3) [μmol/kg-seawater]","pH","Temperature (degrees C)","Salinity (ppt)","Date","DateTime","Year",
                        ],axis = 1)

n = 3 # number of samples per measurement
d['Calcification_Standard_Error_Measurement'] = d['Calcification_Standard Deviation']/np.sqrt(n)
d = d.dropna()
resamples = 5
random_samples = np.random.normal(loc=d['Calcification'].values.reshape(-1,1),scale=d['Calcification_Standard_Error_Measurement'].values.reshape(-1,1), size=(len(d),resamples))
random_samples[random_samples < 0] = 0

# Create a DataFrame from the random samples with new variable names
sample_columns = [f'sample_{i+1}' for i in range(random_samples.shape[1])]
sample_df = pd.DataFrame(random_samples, columns=sample_columns)

# Concatenate the original DataFrame with the new sample DataFrame
d.reset_index(inplace=True)
sample_df.reset_index(inplace=True)
d = pd.concat([d, sample_df], axis=1)
d.drop(["index","Calcification_Standard Deviation","Calcification_Standard_Error_Measurement"],axis=1,inplace=True)

# Grid data to 180x360x41x12 (required for all datasets)
depth_bins = np.linspace(0, 205, 42)
depth_labels = np.linspace(0, 200, 41)
d['Depth'] = pd.cut(d['Depth'], bins=depth_bins, labels=depth_labels).astype(np.float64) 

lat_bins = np.linspace(-90, 90, 181)
lat_labels = np.linspace(-90, 89, 180)
d['Latitude'] = pd.cut(d['Latitude'].astype(np.float64), bins=lat_bins, labels=lat_labels).astype(np.float64) 

lon_bins = np.linspace(-180, 180, 361)
lon_labels = np.linspace(-180, 179, 360)
d['Longitude'] = pd.cut(d['Longitude'].astype(np.float64), bins=lon_bins, labels=lon_labels).astype(np.float64) 

d = d.groupby(['Latitude', 'Longitude', 'Depth', 'Month']).mean().reset_index()
d.rename({'Latitude':'lat','Longitude':'lon','Depth':'depth','Month':'time'},inplace=True,axis=1)
print(d["Calcification"].notna().sum())

# Skip lines 79-100 if you do not want to add pseudo zeros below 0.01% Par (only applicable for CP data)
# Load the 0.01% PAR mask from the NetCDF file
mask_ds = xr.open_dataset('/home/mv23682/Documents/Abil/studies/wiseman2024/env_data_processing/regridded_data/PAR_01prct_mask.nc')

# Assuming the mask is binary (0s and 1s)
mask = mask_ds['mask']

# Select coordinates where the mask is 0
mask_zeros = mask.where(mask == 0, drop=True)

# Convert the selected coordinates to a DataFrame
zeros_df = mask_zeros.to_dataframe().reset_index()

# Drop rows where the mask is not 0
zeros_df = zeros_df[zeros_df['mask'] == 0]

# Randomly select 10% (~10,000 points) for 10:1 pseudo zero to obs
zeros_df_subset = zeros_df.sample(n=(10*(d["Calcification"].notna().sum())), random_state=42)

# Get all relevant columns (Calcification + sample_{n})
columns_to_zero = ['Calcification'] + [col for col in d.columns if col.startswith('sample_')]

# Set the selected columns to zero
zeros_df_subset[columns_to_zero] = 0

# Append the new rows to the original dataframe
d = pd.concat([d, zeros_df_subset[['lat', 'lon', 'depth', 'time'] + columns_to_zero]], ignore_index=True)

## Concat with env data (required for all data)
# Set index for joining to env_data
d.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)
d['dummy'] = 1
print("loading env")

ds = xr.open_dataset('/home/mv23682/Documents/Abil/studies/wiseman2024/data/env_data.nc')
df = ds.to_dataframe()
ds = None 
df.reset_index(inplace=True)
df = df[["temperature","sio4","po4","no3","o2","mld","DIC","TA","PAR","chlor_a","CI_2","time", "depth", "lat", "lon"]]
df.set_index(['lat','lon','depth','time'],inplace=True)

out = pd.concat([d,df], axis=1)
out = out[out["dummy"] == 1]
out = out.drop(['dummy'], axis = 1)
out = out.dropna()
##non_zero_count = out["Calcification"].notna() &
print((out["Calcification"].notna() & (out["Calcification"] != 0)).sum())
out.to_csv("/home/mv23682/Documents/Abil/studies/wiseman2024/data/calcif_env_presample.csv", index=True)

print("fin")
# %%
