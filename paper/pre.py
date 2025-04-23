import xarray as xr 
import pandas as pd

# ds = xr.open_dataset('/home/phyto-2/Abil_SDM_data/env_data.nc')

# ds = ds[["temperature", 
#     "sio4", "po4", "no3", 
#     "o2", "DIC", "TA",
#     "PAR"]]

# ds = ds.isel(depth=1).mean(dim=["time"]) 

# d = ds.to_dataframe()

# d.to_csv("./env_mean_global_surface.csv")


def merge_cascade_env(
        obs_path="../data/gridded_abundances.csv",
        env_path="../data/env_data.nc",
        env_vars=None,
        out_path="../data/obs_env.csv",
        pseudo_absences=1000,
        species=["Emiliania huxleyi HET"]):
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
    species : list of str
        List of species to export to final df

    Returns
    -------
    None
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
    d.rename({'Latitude': 'lat', 'Longitude': 'lon', 'Depth': 'depth', 'Month': 'time'}, inplace=True, axis=1)
    d.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)


    print("Loading environmental data")
    ds = xr.open_dataset(env_path)
    print("Converting environmental data to dataframe")
    df = ds.to_dataframe()
    ds = None
    df.reset_index(inplace=True)
    df.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)
    df = df[env_vars]

    print("Identifying rows in environmental data not present in observational data")
    missing_rows = df.loc[~df.index.isin(d.index)]

    print("Merging observational and environmental data")
    merged = d.merge(df, how="left", left_index=True, right_index=True)
    
    if not missing_rows.empty:
        print(f"Adding {pseudo_absences} pseudo-absences")
        # Sample pseudo-absences
        sampled_na = missing_rows.sample(n=min(pseudo_absences, len(missing_rows)), 
                        replace=len(missing_rows) < pseudo_absences, 
                        random_state=42)
        
        # Initialize species columns with zeros
        for sp in species:
            sampled_na[sp] = 0
        
        merged = pd.concat([merged, sampled_na])

    merged = merged[species+env_vars]
#    merged.dropna(inplace=True)
    merged.to_csv(out_path, index=True)
    print("Finished merging and saving dataset")

# merge_cascade_env(obs_path = "/home/phyto-2/CASCADE/gridded_datasets/gridded_abundances.csv",
#                   env_path= '/home/phyto-2/Abil/studies/devries2024/data/env_data.nc',
#                   env_vars = ["temperature", 
#                             "sio4", "po4", "no3", 
#                             "o2", "DIC", "TA",
#                             "PAR"],
#                     out_path = "/home/phyto-2/Abil/paper/data/training.csv",
#                     pseudo_absences=0,
#                     species=["Emiliania huxleyi HET", "Coccolithus pelagicus HET"])

merge_cascade_env(obs_path = "/home/phyto-2/CASCADE/gridded_datasets/gridded_abundances.csv",
                  env_path= '/home/phyto-2/Abil/studies/devries2024/data/po4_3.nc',
                  env_vars = ["po4"],
                    out_path = "/home/phyto-2/Abil/paper/data/training.csv",
                    pseudo_absences=0,
                    species=["Emiliania huxleyi HET", "Coccolithus pelagicus HET"])
