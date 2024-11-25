import pandas as pd
import numpy as np
import xarray as xr

# Load raw dataset from Marsh et al.
d_raw = pd.read_csv('/home/mv23682/Documents/Abil/studies/wiseman2024/data/calcif_2018_v2.csv',
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

# Drop rows where method is Diff or Ca45 due to quality control concerns
d_filtered = d_raw[~d_raw['Method'].isin(['Diff','Ca45'])]

# Drop rows where the condition "Calcification / Emiliania_huxleyi_cell_counts > 3.5" is met
# Only apply condition where "Emiliania huxleyi cell counts [cells mL-1]" is not NaN
# This removes extraneously high data values that may have quality control concerns
mask = d_filtered["Emiliania huxleyi cell counts [cells mL-1]"].notna() & (
    d_filtered["Calcification"] / d_filtered["Emiliania huxleyi cell counts [cells mL-1]"] > 3.5)
d_filtered = d_filtered[~mask]
d = d_filtered

# Drop data unnecessary for Abil.py
d = d.convert_dtypes()
d = d.drop(["PI","Expedition","OS Region","Reference_Author_Published_year","Reference_doi",
                        "Sample_ID","Irr_Depth",
                        "Optical_Depth","Method","Incubation_Length",
                        "Calcification_Standard Deviation",
                        "Primary_Production_Standard_Deviation",
                        "0.2-2 um Net Primary Production [µmol C m-3 d-1]","0.2-2 um Net Primary Production_Standard Deviation  [µmol C m-3 d-1]",
                        "2-10 um Net Primary Production  [µmol C m-3 d-1]","2-10 um Net Primary Production_Standard Deviation [µmol C m-3 d-1]",
                        ">10 um Net Primary Production  [µmol C m-3 d-1]",">10 um Net Primary Production_Standard Deviation [µmol C m-3 d-1]",
                        "Total Coccolithophore cell counts [cells mL-1]","Emiliania huxleyi cell counts [cells mL-1]",
                        "Chlorophyll-a [mg m-3)","NOx (µM/L)","Silicate (µM/L)","Phosphate (µM/L)","DIC [μmol/kg-seawater]",
                        "Total Alkalinity  [μmol/kg-seawater]","Bicarbonate (HCO3) [μmol/kg-seawater]",
                        "Carbonate (CO3) [μmol/kg-seawater]","pH","Temperature (degrees C)","Salinity (ppt)"
                        ],axis = 1)

# Grid data to 180x360x41x12
depth_bins = np.linspace(0, 205, 42)
depth_labels = np.linspace(0, 200, 41)
d['Depth'] = pd.cut(d['Depth'], bins=depth_bins, labels=depth_labels).astype(np.float64) 

lat_bins = np.linspace(-90, 90, 181)
lat_labels = np.linspace(-90, 89, 180)
d['Latitude'] = pd.cut(d['Latitude'].astype(np.float64), bins=lat_bins, labels=lat_labels).astype(np.float64) 

lon_bins = np.linspace(-180, 180, 361)
lon_labels = np.linspace(-180, 179, 360)
d['Longitude'] = pd.cut(d['Longitude'].astype(np.float64), bins=lon_bins, labels=lon_labels).astype(np.float64) 

d['DateTime'] = pd.to_datetime(d['Date'],dayfirst=True)
d['Month'] = pd.DatetimeIndex(d['DateTime']).month
d['Year'] = pd.DatetimeIndex(d['DateTime']).year

d = d.drop(["Date","DateTime","Year"],axis = 1)
d = d.groupby(['Latitude', 'Longitude', 'Depth', 'Month']).mean().reset_index()
d.rename({'Latitude':'lat','Longitude':'lon','Depth':'depth','Month':'time'},inplace=True,axis=1)


# Skip lines 70-94 if you do not want to add pseudo zeros below 0.01% Par
# Load the 0.01% PAR mask from the NetCDF file
mask_ds = xr.open_dataset('/home/mv23682/Documents/Abil/studies/wiseman2024/data/preprocessing_data/PAR_01prct_mask.nc')

# Assuming the mask is binary (0s and 1s)
mask = mask_ds['mask']

# Select coordinates where the mask is 0
mask_zeros = mask.where(mask == 0, drop=True)

# Convert the selected coordinates to a DataFrame
zeros_df = mask_zeros.to_dataframe().reset_index()

# Drop rows where the mask is not 0
zeros_df = zeros_df[zeros_df['mask'] == 0]

# Randomly select 10% (~30,000 points) for 10:1 pseudo zero to obs
zeros_df_subset = zeros_df.sample(frac=0.1, random_state=42)

# Add columns for zeros in Calcification and Primary_Production
zeros_df_subset['Calcification'] = 0
zeros_df_subset['Primary_Production'] = 0

# Append zeros data to d
d = pd.concat([d, zeros_df_subset[['lat', 'lon', 'depth', 'time', 'Calcification', 'Primary_Production']]], ignore_index=True)

# Set index for joining to env_data
d.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)
d['dummy'] = 1

# Concat with env data
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
out.to_csv("/home/mv23682/Documents/Abil/studies/wiseman2024/data/calcif_env.csv", index=True)

print("fin")