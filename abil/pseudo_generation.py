import pandas as pd
import numpy as np
import xarray as xr
from .analyze import area_of_applicability
from sklearn.preprocessing import StandardScaler


def generate_pseudo_absences(merged_df, missing_rows, env_vars, species_cols, 
                             absence_ratio=1, aoa_threshold=0.99, min_presence=100):
    """
    Generate pseudo-absences for each species at specified ratio to presences.
    
    Parameters:
    - merged_df: DataFrame containing merged observation and environmental data
    - missing_rows: DataFrame containing environmental data without observations
    - env_vars: List of environmental variable names
    - species_cols: List of species column names
    - absence_ratio: Ratio of pseudo-absences to generate relative to presences
    - aoa_threshold: Threshold for Area of Applicability calculation
    - min_presence: Minimum number of presence records required to generate pseudo-absences
    """
    env_feature_vars = [v for v in env_vars if v not in ['time', 'depth', 'lat', 'lon']]
    pseudo_dfs = []
    
    for species in species_cols:
        print(f"\nProcessing species: {species}")
        species_obs = merged_df[merged_df[species].notna()]
        n_presence = len(species_obs)
        
        # Skip if not enough presence records
        if n_presence < min_presence:
            print(f"Only {n_presence} presence records for {species} (minimum {min_presence} required), skipping")
            continue
            
        X_train = species_obs[env_feature_vars].dropna()
        X_predict = missing_rows[env_feature_vars].dropna()
        
        # Skip if no valid training data after dropping NAs
        if len(X_train) < min_presence:
            print(f"No valid environmental data for presence locations of {species}, skipping")
            continue
            
        try:
            # Scale the data before AOA calculation
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_predict_scaled = scaler.transform(X_predict)
            
            # Calculate Area of Applicability on scaled data
            aoa = area_of_applicability(X_predict_scaled, X_train_scaled, feature_weights=False, threshold=aoa_threshold)
            outside_aoa = missing_rows.loc[X_predict.index][aoa == 0]
            
            # Determine number of samples to generate
            n_samples = min(int(n_presence * absence_ratio), len(outside_aoa))
            
            if n_samples == 0:
                print(f"No suitable locations found outside AOA for {species}")
                continue
                
            # Sample pseudo-absences
            sampled_na = outside_aoa.sample(n=n_samples, replace=len(outside_aoa) < n_samples, random_state=42)
            
            # Create species-specific dataframe with pseudo-absences
            species_df = pd.DataFrame({
                s: (0 if s == species else np.nan) for s in species_cols
            }, index=sampled_na.index)
            
            pseudo_dfs.append(pd.concat([sampled_na, species_df], axis=1))
            print(f"Generated {n_samples} pseudo-absences for {species} (presences: {n_presence})")
            
        except Exception as e:
            print(f"Error processing {species}: {str(e)}")
            continue
    
    return pd.concat([merged_df] + pseudo_dfs) if pseudo_dfs else merged_df
