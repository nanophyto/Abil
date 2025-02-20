import pandas as pd
import numpy as np
import xarray as xr
import pickle
import sys


ds = xr.open_dataset('/home/phyto-2/Abil_SDM_data/env_data.nc')
ds.to_netcdf('/home/phyto-2/Abil/studies/devries2024/data/env_data.nc')
ds = None

df =  pd.read_csv('/home/phyto-2/Abil_SDM_data/env_data.csv')
df.to_csv('/home/phyto-2/Abil/studies/devries2024/data/env_data.csv')
df = None

def merge_cascade_env(
        obs_path="../data/gridded_abundances.csv",
        env_path="../data/env_data.nc",
        env_vars=None,
        out_path="../data/obs_env.csv",
        pseudo_absences=1000):
    """
    Merge observational and environmental datasets based on spatial and temporal indices.

    Parameters
    ----------
    obs_path : str, default="../data/gridded_abundances.csv"
        Path to observational data CSV.
    env_path : str, default="../data/env_data.nc"
        Path to environmental data NetCDF file.
    env_vars : list of str, optional
        List of environmental variables to include in the merge.
    out_path : str, default="../data/obs_env.csv"
        Path to save the merged dataset.

    pseudo_absences : int, default=1000
        Number of random pseudo-absences (rows from env data not in obs data) to add to the dataset.

    Returns
    -------
    None
    """
    if env_vars is None:
        env_vars = ["temperature", "sio4", "po4", "no3", "o2", "mld", "DIC",
                    "TA", "irradiance", "chlor_a",
                    "time", "depth", "lat", "lon"]

    # Load observational data
    d = pd.read_csv(obs_path)
    d = d.convert_dtypes()

    # Convert to wide format

    d = d.pivot(index=["Latitude", "Longitude", "Depth", "Month", "Year"],
                columns="Species",
                values="cells L-1").reset_index()

    d = d.groupby(['Latitude', 'Longitude', 'Depth', 'Month']).mean().reset_index()
    d.rename({'Latitude': 'lat', 'Longitude': 'lon', 'Depth': 'depth', 'Month': 'time'}, inplace=True, axis=1)
    d.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)


    print("Loading environmental data")
    ds = xr.open_dataset(env_path)
    print("Converting environmental data to dataframe")
    df = ds.to_dataframe()
    ds = None
    df.reset_index(inplace=True)
    df = df[env_vars]
    df.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)


    print("Identifying rows in environmental data not present in observational data")
    missing_rows = df.loc[~df.index.isin(d.index)]

    print("Merging observational and environmental data")
    merged = d.merge(df, how="left", left_index=True, right_index=True)

    if not missing_rows.empty:
        print(f"Adding {pseudo_absences} pseudo-absences")
        # Sample pseudo-absences
        sampled_na = missing_rows.sample(n=min(pseudo_absences, len(missing_rows)), replace=len(missing_rows) < pseudo_absences, random_state=42)
        merged = pd.concat([merged, sampled_na])

    merged.to_csv(out_path, index=True)
    print("Finished merging and saving dataset")

merge_cascade_env(obs_path = "/home/phyto-2/CASCADE/gridded_datasets/gridded_abundances.csv",
                  env_path= '/home/phyto-2/Abil/studies/devries2024/data/env_data.nc',
                  env_vars = ["temperature", 
                            "sio4", "po4", "no3", 
                            "o2", "mld", "DIC", "TA",
                            "PAR","chlor_a",
                            "time", "depth", 
                            "lat", "lon"],
                    out_path = "/home/phyto-2/Abil/studies/devries2024/data/obs_env.csv",
                    pseudo_absences=100)

# Load the obs_env.csv file
obs = pd.read_csv("/home/phyto-2/Abil/studies/devries2024/data/obs_env.csv")

# Load the summary_table.csv file
d = pd.read_csv("/home/phyto-2/CASCADE/resampled_cellular_datasets/summary_table.csv")
d.rename(columns={"POC (pg poc) [median]": "pg poc", "PIC (pg pic) [median]": "pg pic", "species": "Target"}, inplace=True)
d = d[['Target', 'pg poc', 'pg pic']]

# Initialize an empty list to store non-zero counts for each Target
non_zero_counts = []

# Iterate over each Target in d
for target in d['Target']:
    if target in obs.columns:
        # Subset the column for the target, drop NAs, and count non-zero values
        non_zero_count = obs[target].dropna().ne(0).sum()
    else:
        # If the target is not in obs, assign 0
        non_zero_count = 0
    non_zero_counts.append(non_zero_count)

# Add the non-zero counts as a new column in d
d['n'] = non_zero_counts

d = d[d["n"]>200]

# Save the updated traits.csv file
d.to_csv("/home/phyto-2/Abil/studies/devries2024/data/traits.csv", index=False)
