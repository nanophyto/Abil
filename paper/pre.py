import pandas as pd
import numpy as np
import xarray as xr
from abil.analyze import area_of_applicability
from sklearn.preprocessing import StandardScaler

ds = xr.open_dataset('/home/phyto-2/Abil_SDM_data/env_data.nc')

ds = ds[["temperature", 
    "sio4", "po4", "no3", 
    "o2", "DIC", "TA",
    "PAR"]]

ds = ds.isel(depth=1).mean(dim=["time"]) 

d = ds.to_dataframe()

d.to_csv("./env_mean_global_surface.csv")

def generate_pseudo_absences(merged_df, missing_rows, env_vars, species_cols, 
                             absence_ratio=3, aoa_threshold=0.99, min_presence=100):
    """
    Generate pseudo-absences for specified species at specified ratio to presences.
    
    Parameters:
    - merged_df: DataFrame containing merged observation and environmental data
    - missing_rows: DataFrame containing environmental data without observations
    - env_vars: List of environmental variable names
    - species_cols: List of species column names to process (can be a subset of all species in data)
    - absence_ratio: Ratio of pseudo-absences to generate relative to presences
    - aoa_threshold: Threshold for Area of Applicability calculation
    - min_presence: Minimum number of presence records required to generate pseudo-absences
    """
    env_feature_vars = [v for v in env_vars if v not in ['time', 'depth', 'lat', 'lon']]
    pseudo_dfs = []
    
    # Get all available species columns in case some requested ones don't exist
    available_species = [col for col in merged_df.columns if col not in env_vars]
    species_to_process = [s for s in species_cols if s in available_species]
    
    if not species_to_process:
        print("Warning: None of the requested species were found in the data")
        return merged_df
    
    # First filter missing_rows to only complete cases
    complete_missing_rows = missing_rows.dropna(subset=env_feature_vars)
    
    for species in species_to_process:
        print(f"\nProcessing species: {species}")
        species_obs = merged_df[merged_df[species].notna()]
        n_presence = len(species_obs)
        
        # Skip if not enough presence records
        if n_presence < min_presence:
            print(f"Only {n_presence} presence records for {species} (minimum {min_presence} required), skipping")
            continue
            
        # Ensure we only use complete cases for training
        X_train = species_obs[env_feature_vars].dropna()
        
        # Skip if no valid training data after dropping NAs
        if len(X_train) < 100:
            print(f"Not enough valid environmental data for presence locations of {species} ({len(X_train)} records), skipping")
            continue
            
        try:
            # Scale the data before AOA calculation
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_predict_scaled = scaler.transform(complete_missing_rows[env_feature_vars])
            
            # Calculate Area of Applicability on scaled data
            aoa = area_of_applicability(X_predict_scaled, X_train_scaled, feature_weights=False, threshold=aoa_threshold)
            outside_aoa = complete_missing_rows[aoa == 1]
            
            # Determine number of samples to generate
            n_samples = min(n_presence * absence_ratio, len(outside_aoa))
            
            if n_samples == 0:
                print(f"No suitable locations found outside AOA for {species}")
                continue
                
            # Sample pseudo-absences
            sampled_na = outside_aoa.sample(n=n_samples, replace=len(outside_aoa) < n_samples, random_state=42)
            
            # Create species-specific dataframe with pseudo-absences
            species_df = pd.DataFrame({
                s: (0 if s == species else np.nan) for s in species_to_process
            }, index=sampled_na.index)
            
            pseudo_dfs.append(pd.concat([sampled_na, species_df], axis=1))
            print(f"Generated {n_samples} pseudo-absences for {species} (presences: {n_presence})")
            
        except Exception as e:
            print(f"Error processing {species}: {str(e)}")
            continue
    
    return pd.concat([merged_df] + pseudo_dfs) if pseudo_dfs else merged_df

def merge_cascade_env(
        obs_path="../data/gridded_abundances.csv",
        env_path="../data/env_data.nc",
        env_vars=None,
        species_list=None,  # New parameter for specific species
        out_path="../data/obs_env.csv",
        absence_ratio=1):  # 0 means no pseudo-absences
    """
    Merge observational and environmental datasets based on spatial and temporal indices.
    
    Parameters:
    - obs_path: Path to observational data CSV
    - env_path: Path to environmental data NetCDF
    - env_vars: List of environmental variables to include
    - species_list: List of specific species to process (None processes all species)
    - out_path: Output file path
    - absence_ratio: Ratio of pseudo-absences to generate
    """
    if env_vars is None:
        raise ValueError("env_vars not defined")

    # Load observational data
    d = pd.read_csv(obs_path)
    d = d.convert_dtypes()

    # Convert to wide format
    d = d.pivot(index=["Latitude", "Longitude", "Depth", "Month", "Year"],
                columns="Species",
                values="cells L-1").reset_index()

    d = d.groupby(['Latitude', 'Longitude', 'Depth', 'Month']).mean().reset_index()
    d.rename({'Latitude': 'lat', 'Longitude': 'lon', 'Depth': 'depth', 'Month': 'time'}, 
             inplace=True, axis=1)
    d.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)
    d = d.drop('Year', axis=1)

    # Load environmental data
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

    if not missing_rows.empty and absence_ratio > 0:
        # Use specified species list or all available species
        species_cols = species_list if species_list is not None else [col for col in d.columns if col not in env_vars]
        merged = generate_pseudo_absences(
            merged_df=merged,
            missing_rows=missing_rows,
            env_vars=env_vars,
            species_cols=species_cols,
            absence_ratio=absence_ratio
        )

    merged['Gephyrocapsa huxleyi HET'] = merged['Emiliania huxleyi HET']

    merged.reset_index(inplace=True)

    merged = merged[['Gephyrocapsa huxleyi HET', 'lat', 'lon', 'temperature', 'sio4', 'po4', 'no3', 'o2', 'DIC', 'TA', 'PAR']]
    # Create a mask for rows to KEEP (i.e., NOT (latitude > 40 AND longitude between 120-180))
    mask = ~((merged['lat'] > 40) & (merged['lon'] >= 120) & (merged['lon'] <= 180))
    # Filter the DataFrame
    merged = merged[mask]

    merged = merged.dropna(subset=["Gephyrocapsa huxleyi HET"])

    merged.to_csv(out_path, index=False)
    print("Finished merging and saving dataset")

# Example usage with specific species:
merge_cascade_env(
    obs_path="/home/phyto-2/CASCADE/gridded_datasets/gridded_abundances.csv",
    env_path='/home/phyto-2/Abil_SDM_data/env_data.nc',
    env_vars=["temperature", "sio4", "po4", "no3", "o2", "DIC", "TA", "PAR", "time", "depth", "lat", "lon"],
    species_list=["Emiliania huxleyi HET"],  # Specify species to process here
    out_path="/home/phyto-2/Abil/paper/data/training.csv",
    absence_ratio=1
)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import numpy as np

d = pd.read_csv("/home/phyto-2/Abil/paper/data/training.csv")

d['Gephyrocapsa huxleyi HET'] = np.log10(d['Gephyrocapsa huxleyi HET'] + 1)

# --- Plot Training Data ---
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()  # This ensures the whole world is shown
ax.coastlines()  # Add coastlines

# Create the scatter plot
sc = ax.scatter(d['lon'], d['lat'], c=d['Gephyrocapsa huxleyi HET'],
               cmap='viridis', s=10, transform=ccrs.PlateCarree(),
               vmin=0)

# Add colorbar
plt.colorbar(sc, label='Log10(Gephyrocapsa huxleyi HET + 1)')

# Set title
plt.title('Distribution of Gephyrocapsa huxleyi HET (Training Data)')

plt.show()