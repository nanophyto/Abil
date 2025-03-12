import numpy as np
import pandas as pd
import xarray as xr

from sklearn.datasets import make_regression
from sklearn.utils import resample
from sklearn.metrics import roc_curve, roc_auc_score

from joblib import delayed
import warnings
from xgboost import DMatrix

def _predict_one_member(i, member, chunk):
    """
    """
    with warnings.catch_warnings(
        action='ignore', category=UserWarning
    ):
        try:
            return member.predict(DMatrix(chunk), iteration_range=(i, i+1))
        except TypeError:
            return member.predict(chunk)


def upsample(d, target, ratio=10):
    """
    Upsample zero and non-zero observations in the dataset to balance classes.
    
    Parameters
    ----------
    d : pd.DataFrame
        Input dataframe.
    target : str
        Target column for upsampling.
    ratio : int, default=10
        Ratio of zeros to non-zero samples after upsampling.

    Returns
    -------
    ix : pd.DataFrame
        Upsampled dataframe.
    """
    # Separate presence (non-zero) and absence (zero) indices
    presence_indices = d[d[target] != 0].index
    absence_indices = d[d[target] == 0].index

    # Calculate the number of zero samples to match the ratio
    num_presence = len(presence_indices)
    num_absence_to_sample = num_presence * ratio

    # Sample zeros with replacement if needed
    sampled_absence_indices = np.random.choice(absence_indices, size=num_absence_to_sample, replace=True)

    # Create the upsampled dataframe by combining presences and sampled absences
    upsampled_df = pd.concat([d.loc[presence_indices], d.loc[sampled_absence_indices]], ignore_index=True)

    return upsampled_df


def merge_obs_env(obs_path="../data/gridded_abundances.csv",
                  env_path="../data/env_data.nc",
                  env_vars=None,
                  out_path="../data/obs_env.csv"):
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

    Returns
    -------
    None
    """
    if env_vars is None:
        env_vars = ["temperature", "sio4", "po4", "no3", "o2", "mld", "DIC",
                    "TA", "irradiance", "chlor_a", "Rrs_547", "Rrs_667", "CI_2",
                    "time", "depth", "lat", "lon"]

    d = pd.read_csv(obs_path)
    d = d.convert_dtypes()
    d = d.groupby(['Latitude', 'Longitude', 'Depth', 'Month']).mean().reset_index()
    d.rename({'Latitude': 'lat', 'Longitude': 'lon', 'Depth': 'depth', 'Month': 'time'}, inplace=True, axis=1)
    d.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)

    print("loading env")
    ds = xr.open_dataset(env_path)
    print("converting to dataframe")
    df = ds.to_dataframe()
    ds = None
    df.reset_index(inplace=True)
    df = df[env_vars]
    df.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)
    print("merging environment")
    out = d.merge(df, how="left", left_index=True, right_index=True)
    out.to_csv(out_path, index=True)
    print("fin")

def example_data(
    y_name, 
    n_samples=100, 
    n_features=5, 
    noise=0.1, 
    train_to_predict_ratio=0.7, 
    zero_to_non_zero_ratio=0.5, 
    random_state=59
    ):
    """
    Generate training and prediction datasets with ['lat', 'lon', 'depth', 'time'] indices.
    Includes zeros in the target and allows upsampling of zero values.
    
    Parameters:
        y_name (str): Name of the target variable.
        n_samples (int): Total number of samples to generate (training + prediction).
        n_features (int): Number of features for the dataset.
        noise (float): Noise level for the regression data.
        train_to_predict_ratio (float): Ratio of training to prediction data.
        zero_to_non_zero_ratio (float): Ratio of zero to non-zero target values after upsampling.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train (pd.DataFrame): Training feature dataset with MultiIndex.
        X_predict (pd.DataFrame): Prediction feature dataset with MultiIndex.
        y (pd.Series): Target variable for training dataset.
    """
    np.random.seed(random_state)  # Set random seed for reproducibility
    
    # Generate regression data
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    
    # Introduce zeros randomly in the target variable
    zero_fraction = 0.2  # Fraction of zero values to introduce
    zero_indices = np.random.choice(len(y), size=int(len(y) * zero_fraction), replace=False)
    y[zero_indices] = 0
    
    # Ensure target values are non-negative by taking absolute values
    y = np.abs(y)
    
    # Split data into training and prediction sets
    train_size = int(n_samples * train_to_predict_ratio)
    X_train, X_predict = X[:train_size], X[train_size:]
    y_train = y[:train_size]
    
    # Upsample zeros in the training set
    X_train_df = pd.DataFrame(X_train, columns=[f"feature_{i+1}" for i in range(X_train.shape[1])])
    y_train_series = pd.Series(y_train, name=y_name)
    
    zero_mask = y_train_series == 0
    non_zero_mask = y_train_series > 0
    
    X_zeros = X_train_df[zero_mask]
    y_zeros = y_train_series[zero_mask]
    X_non_zeros = X_train_df[non_zero_mask]
    y_non_zeros = y_train_series[non_zero_mask]
    
    # Upsample zeros to achieve the desired zero_to_non_zero_ratio
    upsample_count = int(len(y_non_zeros) * zero_to_non_zero_ratio)
    X_zeros_upsampled, y_zeros_upsampled = resample(
        X_zeros, y_zeros, 
        replace=True, 
        n_samples=upsample_count, 
        random_state=random_state
    )
    
    # Combine upsampled data
    X_train_combined = pd.concat([X_non_zeros, X_zeros_upsampled], axis=0)
    y_train_combined = pd.concat([y_non_zeros, y_zeros_upsampled], axis=0)
    
    # Shuffle combined data
    combined_indices = np.random.permutation(X_train_combined.index)
    X_train_combined = X_train_combined.loc[combined_indices]
    y_train_combined = y_train_combined.loc[combined_indices]
    
    # Generate random latitude, longitude, depth, and time
    latitudes_train = np.round(np.random.uniform(-90, 90, size=X_train_combined.shape[0]), -1)
    longitudes_train = np.round(np.random.uniform(-180, 180, size=X_train_combined.shape[0]), -1)
    depths_train = np.round(np.random.uniform(0, 200, size=X_train_combined.shape[0]), -1)
    times_train = np.round(np.random.randint(1, 13, size=X_train_combined.shape[0]), -1)
    
    latitudes_predict = np.round(np.random.uniform(-90, 90, size=X_predict.shape[0]), -1)
    longitudes_predict = np.round(np.random.uniform(-180, 180, size=X_predict.shape[0]), -1)
    depths_predict = np.round(np.random.uniform(0, 200, size=X_predict.shape[0]), -1)
    times_predict = np.round(np.random.randint(1, 13, size=X_predict.shape[0]), -1)
    
    # Set MultiIndex for X_train and X_predict
    X_train_combined.index = pd.MultiIndex.from_arrays(
        [latitudes_train, longitudes_train, depths_train, times_train],
        names=['lat', 'lon', 'depth', 'time']
    )
    X_predict = pd.DataFrame(X_predict, columns=[f"feature_{i+1}" for i in range(X_predict.shape[1])])
    X_predict.index = pd.MultiIndex.from_arrays(
        [latitudes_predict, longitudes_predict, depths_predict, times_predict],
        names=['lat', 'lon', 'depth', 'time']
    )
    
    # Set y_train index
    y_train_combined.index = X_train_combined.index
    
    return X_train_combined, X_predict, y_train_combined


def abbreviate_species(species_name):
    """
    Abbreviate a species name by shortening the first word to its initial.

    Parameters
    ----------
    species_name : str
        Full species name.

    Returns
    -------
    str
        Abbreviated species name.
    """
    words = species_name.split()
    if len(words) == 1:
        return species_name
    abbreviated_name = words[0][0].upper() + '.'
    abbreviated_name += ' ' + ' '.join(words[1:])
    return abbreviated_name


def inverse_weighting(values):
    """
    Compute inverse weighting for a list of values.

    Parameters
    ----------
    values : list of float
        Input values.

    Returns
    -------
    list of float
        Normalized inverse weights.
    """
    inverse_weights = [1 / value for value in values]
    total_inverse_weight = sum(inverse_weights)
    normalized_weights = [weight / total_inverse_weight for weight in inverse_weights]
    return normalized_weights


def find_optimal_threshold(model, X, y_test):
    """
    Finds the optimal probability threshold for binary classification using the ROC curve and Youden's Index.

    Parameters:
    -----------
    model : sklearn classifier
        A fitted binary classification model
    X : array-like of shape (n_samples, n_features)
        Input features for the test or validation set.
    y_test : array-like of shape (n_samples,)
        True binary labels for the test or validation set.

    Returns:
    --------
    optimal_threshold : float
        The optimal probability threshold for classifying a sample as present.
    """
    
    # Get predicted probabilities for the positive class
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Calculate optimal threshold using Youden's Index
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold