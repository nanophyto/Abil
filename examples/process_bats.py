import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import pandas as pd
import numpy as np

# ---- Constants for BATS ----
BATS_LAT = 31.667
BATS_LON = -64.167   # west is negative in [-180, 180]

def _normalize_lon_to_180(lon):
    return ((lon + 180) % 360) - 180

def _find_depth_name(ds_or_df):
    # Common depth names to accept
    candidates = ["depth", "Depth", "DEPTH", "z", "lev"]
    if isinstance(ds_or_df, xr.Dataset):
        for c in candidates:
            if c in ds_or_df.dims or c in ds_or_df.coords:
                return c
    else:  # DataFrame
        for c in candidates:
            if c in ds_or_df.columns:
                return c
    raise ValueError("Could not find a depth column/dimension among: " + ", ".join(candidates))

def preprocess_bats(
    data,
    lat_col="lat",
    lon_col="lon",
    window_deg=0.5,
    bin_size=25.0,
):
    """
    Subset a box around BATS, BIN DEPTH into fixed layers (25 m by default),
    and average over lat/lon so the result keeps only (time, depth).

    Depths are returned as negative (surface = 0, deeper = negative).

    Parameters
    ----------
    data : xr.Dataset or pd.DataFrame
        Input data.
    lat_col, lon_col : str
        Column names for DataFrame inputs.
    window_deg : float
        Half-width of the BATS box in degrees.
    bin_size : float
        Depth bin size in meters.

    Returns
    -------
    xr.Dataset or pd.DataFrame
        Averaged over lat/lon with depth binned. Remaining dims/columns are
        typically ('time','depth').
    """
    # -------- xarray case --------
    if isinstance(data, xr.Dataset):
        ds = data

        if "lon" not in ds.coords or "lat" not in ds.coords:
            raise ValueError("xarray.Dataset must have 'lat' and 'lon' coordinates.")
        depth_name = _find_depth_name(ds)

        # Normalize longitude if needed
        if float(ds.lon.max()) > 180:
            ds = ds.assign_coords(lon=_normalize_lon_to_180(ds.lon)).sortby("lon")

        # Subset to BATS box
        lat_min, lat_max = BATS_LAT - window_deg, BATS_LAT + window_deg
        lon_min, lon_max = BATS_LON - window_deg, BATS_LON + window_deg
        ds_box = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

        # ---- Depth binning ----
        if depth_name not in ds_box.dims and depth_name not in ds_box.coords:
            raise ValueError(f"Depth coordinate '{depth_name}' not found after subsetting.")

        depth_vals = ds_box[depth_name].values
        dmin = float(np.nanmin(depth_vals))
        dmax = float(np.nanmax(depth_vals))
        start = bin_size * np.floor(dmin / bin_size)
        stop = bin_size * np.ceil(dmax / bin_size) + bin_size
        edges = np.arange(start, stop + 1e-9, bin_size)
        centers = (edges[:-1] + edges[1:]) / 2.0

        ds_binned = (
            ds_box
            .groupby_bins(ds_box[depth_name], bins=edges, labels=centers, right=False)
            .mean(dim=depth_name, skipna=True, keep_attrs=True)
        )

        # Rename bins dim to "depth" and make depths negative
        bins_dim = f"{depth_name}_bins"
        ds_binned = ds_binned.rename({bins_dim: "depth"})
        ds_binned = ds_binned.assign_coords(depth=("depth", -centers))

        # ---- Average over lat/lon AFTER depth binning ----
        reduce_dims = [d for d in ("lat", "lon") if d in ds_binned.dims]
        ds_avg = ds_binned.mean(dim=reduce_dims, skipna=True, keep_attrs=True)

        # Order (time, depth, ...)
        desired = [d for d in ("time", "depth") if d in ds_avg.dims]
        other = [d for d in ds_avg.dims if d not in desired]
        ds_avg = ds_avg.transpose(*desired, *other)

        return ds_avg

    # -------- pandas case --------
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        if lon_col not in df.columns or lat_col not in df.columns:
            raise ValueError(f"DataFrame must have '{lat_col}' and '{lon_col}' columns.")
        depth_col = _find_depth_name(df)

        # Normalize lon
        df[lon_col] = _normalize_lon_to_180(df[lon_col].astype(float))

        # Subset to BATS box
        lat_min, lat_max = BATS_LAT - window_deg, BATS_LAT + window_deg
        lon_min, lon_max = BATS_LON - window_deg, BATS_LON + window_deg
        sel = df[lat_col].between(lat_min, lat_max) & df[lon_col].between(lon_min, lon_max)
        df = df.loc[sel].copy()
        if df.empty:
            raise ValueError("No rows inside the BATS box. Try increasing window_deg.")

        # ---- Depth binning ----
        dmin = float(np.nanmin(df[depth_col].to_numpy()))
        dmax = float(np.nanmax(df[depth_col].to_numpy()))
        start = bin_size * np.floor(dmin / bin_size)
        stop = bin_size * np.ceil(dmax / bin_size) + bin_size
        edges = np.arange(start, stop + 1e-9, bin_size)
        centers = (edges[:-1] + edges[1:]) / 2.0

        df["depth"] = pd.cut(
            df[depth_col].astype(float),
            bins=edges,
            right=False,
            labels=-centers,   # <-- make depth negative
            include_lowest=True,
        ).astype(float)

        # Average over lat/lon, grouped by (time, depth)
        group_cols = [c for c in ("time", "depth") if c in df.columns]
        value_cols = [c for c in df.columns if c not in group_cols + [lat_col, lon_col, depth_col]]

        out = (
            df.groupby(group_cols, dropna=False, as_index=False)[value_cols]
              .mean(numeric_only=True)
        )

        # Ensure canonical column order
        lead = [c for c in ("time", "depth") if c in out.columns]
        out = out[lead + [c for c in out.columns if c not in lead]]

        return out

    else:
        raise TypeError("Input must be an xarray.Dataset or pandas.DataFrame")

ds = xr.open_dataset("/home/phyto-2/coccolithophore_sdm/model/data/env_data.nc")
prediction = preprocess_bats(ds, window_deg=0.5)
prediction['temperature'].plot()
plt.show()


df = pd.read_csv(os.path.join("/home/phyto-2/coccolithophore_sdm/model/data/obs_env_1-1.csv"))
training = preprocess_bats(df, window_deg=5)

training = training.rename(columns={"Emiliania huxleyi HET":"Gephyrocapsa huxleyi", "phosphate": "po4", "din": "no3", "irradiance":"PAR", "silicate":"sio4"})
print(training.head())
training = training[['time', 'depth', 'Gephyrocapsa huxleyi', "temperature", "po4","no3", "o2", "PAR", "DIC", "TA"]]
training.dropna(inplace=True)
training.to_csv("/home/phyto-2/Abil/docs/data/bats_training.csv", index=False)

prediction  = prediction.to_dataframe()
prediction.dropna(inplace=True)
prediction.to_csv("/home/phyto-2/Abil/docs/data/bats_prediction.csv")

