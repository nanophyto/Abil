import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .analyze import area_of_applicability


def generate_pseudo_absences(
    merged_df,
    missing_rows,
    env_vars,
    species_cols,
    absence_ratio=1,
    aoa_threshold=0.99,
    min_presence=100,
    allow_replacement=True,
):
    """
    Generate pseudo-absences for each species at a specified ratio to presences.

    Pseudo-absences are sampled from rows without observations that fall outside
    the area of applicability estimated from each species' observed rows.
    ``area_of_applicability`` uses ``0`` for inside AOA and ``1`` for
    outside AOA, so this function samples candidates where the AOA mask
    equals ``1``.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        Merged observation and environmental data.
    missing_rows : pandas.DataFrame
        Environmental rows without observations that can be sampled as
        pseudo-absence candidates.
    env_vars : list of str
        Environmental variable names. Coordinate variables named ``time``,
        ``depth``, ``lat``, and ``lon`` are excluded from the AOA feature set.
    species_cols : list of str
        Species column names.
    absence_ratio : float, default=1
        Number of pseudo-absences to sample relative to the number of presences.
    aoa_threshold : float or str, default=0.99
        Threshold passed to :func:`abil.analyze.area_of_applicability`.
    min_presence : int, default=100
        Minimum number of presence records required to generate pseudo-absences
        for a species.
    allow_replacement : bool, default=True
        If True, sample outside-AOA candidate rows with replacement when there
        are fewer available candidate rows than requested by ``absence_ratio``.
        This allows ``absence_ratio=1`` to produce one pseudo-absence per
        presence even when outside-AOA candidates are scarce. If False, the
        number of pseudo-absences is capped by the number of unique outside-AOA
        candidate rows.

    Returns
    -------
    pandas.DataFrame
        ``merged_df`` plus sampled pseudo-absence rows. If no pseudo-absences
        can be generated, a copy of ``merged_df`` is returned.
    """
    if absence_ratio < 0:
        raise ValueError("absence_ratio must be non-negative")
    if min_presence < 1:
        raise ValueError("min_presence must be at least 1")

    missing = [col for col in env_vars + species_cols if col not in merged_df.columns and col not in missing_rows.columns]
    if missing:
        raise KeyError(f"columns not found in merged_df or missing_rows: {missing}")

    env_feature_vars = [v for v in env_vars if v not in ["time", "depth", "lat", "lon"]]
    if not env_feature_vars:
        raise ValueError("env_vars must contain at least one non-coordinate environmental variable")

    missing_env = [col for col in env_feature_vars if col not in missing_rows.columns]
    if missing_env:
        raise KeyError(f"environmental columns not found in missing_rows: {missing_env}")

    pseudo_dfs = []

    for species in species_cols:
        if species not in merged_df.columns:
            raise KeyError(f"species column not found in merged_df: {species}")

        species_obs = merged_df[merged_df[species].notna()]
        n_presence = len(species_obs)

        if n_presence < min_presence:
            continue

        X_train = species_obs[env_feature_vars].dropna()
        X_predict = missing_rows[env_feature_vars].dropna()

        if len(X_train) < min_presence or X_predict.empty:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_predict_scaled = scaler.transform(X_predict)

        aoa = area_of_applicability(
            X_predict_scaled,
            X_train_scaled,
            feature_weights=False,
            threshold=aoa_threshold,
        )[0]

        # area_of_applicability returns 0 for inside AOA and 1 for outside AOA.
        outside_aoa = missing_rows.loc[X_predict.index][np.asarray(aoa) == 1]

        n_samples = int(n_presence * absence_ratio)
        if n_samples == 0 or outside_aoa.empty:
            continue

        if not allow_replacement:
            n_samples = min(n_samples, len(outside_aoa))

        sampled_na = outside_aoa.sample(
            n=n_samples,
            replace=allow_replacement and len(outside_aoa) < n_samples,
            random_state=42,
        )
        species_df = pd.DataFrame(
            {s: (0 if s == species else np.nan) for s in species_cols},
            index=sampled_na.index,
        )
        pseudo_dfs.append(pd.concat([sampled_na, species_df], axis=1))

    if pseudo_dfs:
        return pd.concat([merged_df] + pseudo_dfs)
    return merged_df.copy()
