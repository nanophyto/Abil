import numpy as np
import pandas as pd
import xarray as xr
from inspect import signature

from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor, is_classifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer
from sklearn.utils import resample


def do_log(self, x):
    y = np.log(x+1)
    return(y)

def do_exp(self, x):
    y = np.exp(x)-1
    return(y)

def upsample(d, target, ratio=10):
        
    y_binary = np.where(d[target]!=0, 1, 0)

    nix = np.where(y_binary==0)[0] #absence index
    pix = np.where(y_binary==1)[0] #presence index

    pixu = d.iloc[pix].sample(pix.shape[0], replace=True)
    nixu = d.iloc[nix].sample(pix.shape[0]*ratio, replace=True)

    ix = pd.concat([pixu, nixu], ignore_index=True)

    return(ix)

def merge_obs_env(obs_path = "../data/gridded_abundances.csv",
                  env_path = "../data/env_data.nc",
                  env_vars = ["temperature", "si", 
                              "phosphate", "din", 
                              "o2", "mld", "DIC", 
                              "TA", "irradiance", 
                              "chlor_a","Rrs_547",
                              "Rrs_667","CI_2",
                              "pic","si_star",
                              "si_n","FID", 
                              "time", "depth", 
                              "lat", "lon"],
                    out_path = "../data/obs_env.csv"):

    d = pd.read_csv(obs_path)

    d = d.convert_dtypes()

    d = d.groupby(['Latitude', 'Longitude', 'Depth', 'Month']).mean().reset_index()
    d.rename({'Latitude':'lat','Longitude':'lon','Depth':'depth','Month':'time'},inplace=True,axis=1)
    d.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)

    print("loading env")

    ds = xr.open_dataset(env_path)
    print("converting to dataframe")
    df = ds.to_dataframe()
    ds = None 
    df.reset_index(inplace=True)
    df = df[env_vars]
    df.set_index(['lat','lon','depth','time'],inplace=True)
    print("merging environment")

    out = d.merge(df, how="left", left_index=True, right_index=True)
    out.to_csv(out_path, index=True)
    print("fin")

class ZeroInflatedRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, classifier, regressor) -> None:
        """Initialize."""
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data.

        y : np.ndarray, 1-dimensional
            The target values.

        sample_weight : Optional[np.array], default=None
            Individual weights for each sample.

        Returns
        -------
        ZeroInflatedRegressor
            Fitted regressor.

        Raises
        ------
        ValueError
            If `classifier` is not a classifier or `regressor` is not a regressor.
        """
        if not is_classifier(self.classifier):
            raise ValueError(
                f"`classifier` has to be a classifier. Received instance of {type(self.classifier)} instead.")
        if not is_regressor(self.regressor):
            raise ValueError(f"`regressor` has to be a regressor. Received instance of {type(self.regressor)} instead.")

        try:
            check_is_fitted(self.classifier)
            self.classifier_ = self.classifier
        except NotFittedError:
            self.classifier_ = clone(self.classifier)

            self.classifier_.fit(X, y != 0)

        non_zero_indices = np.where(self.classifier_.predict(X) == 1)[0]

        if non_zero_indices.size > 0:
            try:
                check_is_fitted(self.regressor)
                self.regressor_ = self.regressor
            except NotFittedError:
                self.regressor_ = clone(self.regressor)

                if isinstance(X, pd.DataFrame):
                    self.regressor_.fit(
                            X.iloc[non_zero_indices],
                            y[non_zero_indices],
                    )
                else:

                    self.regressor_.fit(
                            X[non_zero_indices],
                            y[non_zero_indices],
                    )
        else:
            print("all predictions are zero (!)")
            try:
                check_is_fitted(self.regressor)
                self.regressor_ = self.regressor
            except NotFittedError:
                print("regressor has also not been fitted (!)")
                self.regressor_ = clone(self.regressor)

        return self


    def predict(self, X):
        """
        Get predictions.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples to get predictions of.

        Returns
        -------
        y : np.ndarray, shape (n_samples,)
            The predicted values.
        """
        output = np.zeros(len(X))

        non_zero_indices = np.where(self.classifier_.predict(X) == 1)[0]

        if non_zero_indices.size > 0:
            if isinstance(X, pd.DataFrame):
                output[non_zero_indices] = self.regressor_.predict(X.iloc[non_zero_indices])
            else:
                output[non_zero_indices] = self.regressor_.predict(X[non_zero_indices])

        return output


class LogGridSearch:
    def __init__(self, m, verbose, cv, param_grid, scoring, regions=None):
        self.m = m
        self.verbose = verbose
        self.cv = cv
        self.param_grid = param_grid
        self.scoring = scoring
        self.regions = regions

    def do_log(self, x):
        y = np.log(x+1)
        return(y)
    
    def do_exp(self, x):
        y = np.exp(x)-1
        return(y)
    
    def transformed_fit(self, X, y, log, predictors):

        if log=="yes":

            model = TransformedTargetRegressor(self.m, func = self.do_log, inverse_func=self.do_exp)
            grid_search = GridSearchCV(model, param_grid = self.param_grid, scoring=self.scoring, refit="MAE",
                            cv = self.cv, verbose = self.verbose, return_train_score=True, error_score=-1e99)
            grid_search.fit(X, y)
        
        elif log=="no":

            model = TransformedTargetRegressor(self.m, func = None, inverse_func=None)
            grid_search = GridSearchCV(model, param_grid = self.param_grid, scoring=self.scoring, refit="MAE",
                            cv = self.cv, verbose = self.verbose, return_train_score=True, error_score=-1e99)
            grid_search.fit(X, y)

        elif log =="both":

            normal_m = TransformedTargetRegressor(self.m, func = None, inverse_func=None)
            grid_search1 = GridSearchCV(normal_m, param_grid = self.param_grid, scoring=self.scoring, refit="MAE",
                            cv = self.cv, verbose = self.verbose, return_train_score=True, error_score=-1e99)
            grid_search1.fit(X, y)

            log_m = TransformedTargetRegressor(self.m, func = self.do_log, inverse_func=self.do_exp)
            grid_search2 = GridSearchCV(log_m, param_grid = self.param_grid, scoring=self.scoring, refit="MAE",
                            cv = self.cv, verbose = self.verbose, return_train_score=True, error_score=-1e99)
            grid_search2.fit(X, y)

            if (grid_search1.best_score_ > grid_search2.best_score_):
                grid_search = grid_search1
                best_transformation = "nolog"
                print("best = nolog")
        
            elif (grid_search1.best_score_ < grid_search2.best_score_):
                grid_search = grid_search2
                best_transformation = "log"
                print("best = log")
            else:
                print("same performance for both models")
                grid_search = grid_search1
                best_transformation = "nobest"
            grid_search.cv_results_['best_transformation'] = best_transformation

        else: 
            print("defined log invalid, pick from yes, no or both")

        return grid_search


class UpsampledZeroStratifiedKFold:
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        #convert target variable to binary for stratified sampling
        y_binary = np.where(y!=0, 1, 0)

        for rx, tx in StratifiedKFold(n_splits=self.n_splits).split(X,y_binary):
            nix = np.where(y_binary[rx]==0)[0]
            pix = np.where(y_binary[rx]==1)[0]
            pixu = np.random.choice(pix, size=nix.shape[0], replace=True)
            ix = np.append(nix, pixu)
            rxm = rx[ix]
            yield rxm, tx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
    
    
    
class ZeroStratifiedKFold:
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        #convert target variable to binary for stratified sampling
        y_binary = np.where(y!=0, 1, 0)

        # Check if there are any zeros in the array
        if any(element == 0 for element in y_binary):
            for rx, tx in StratifiedKFold(n_splits=self.n_splits).split(X,y_binary):
                yield rx, tx
        else:
            for rx, tx in KFold(n_splits=self.n_splits).split(X,y_binary):
                yield rx, tx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
    
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

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
    latitudes_train = np.random.uniform(-90, 90, size=X_train_combined.shape[0])
    longitudes_train = np.random.uniform(-180, 180, size=X_train_combined.shape[0])
    depths_train = np.random.uniform(0, 200, size=X_train_combined.shape[0])
    times_train = np.random.randint(1, 13, size=X_train_combined.shape[0])
    
    latitudes_predict = np.random.uniform(-90, 90, size=X_predict.shape[0])
    longitudes_predict = np.random.uniform(-180, 180, size=X_predict.shape[0])
    depths_predict = np.random.uniform(0, 200, size=X_predict.shape[0])
    times_predict = np.random.randint(1, 13, size=X_predict.shape[0])
    
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


# def example_training_data(y_name, n_samples=100, n_features=5, noise=0.1, random_state=59):
    
#     X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
#     X_train = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])

#     # Generate random latitude and longitude and set as MultiIndex
#     latitudes = np.random.uniform(-90, 90, size=X_train.shape[0])
#     longitudes = np.random.uniform(-180, 180, size=X_train.shape[0])
#     X_train.index = pd.MultiIndex.from_tuples(zip(latitudes, longitudes), names=['Latitude', 'Longitude'])

#     #convert y to pandas and define name:
#     y = pd.Series(y)
#     y.name = y_name
#     return(X_train, y)



# def example_predict_data(n_samples=100, n_features=5, noise=0.1, random_state=59):

#     # Generate new sample data for X_predict (this is the data for which we do not have y)
#     X_predict, _ = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
#     X_predict = pd.DataFrame(X_predict, columns=[f"feature_{i+1}" for i in range(X_predict.shape[1])])

#     # Generate random latitude and longitude and set as MultiIndex for X_predict
#     latitudes_predict = np.random.uniform(-90, 90, size=X_predict.shape[0])
#     longitudes_predict = np.random.uniform(-180, 180, size=X_predict.shape[0])
#     X_predict.index = pd.MultiIndex.from_tuples(zip(latitudes_predict, longitudes_predict), names=['Latitude', 'Longitude'])

#     return(X_predict)



def abbreviate_species(species_name):
    words = species_name.split()
    if len(words) == 1:
        return species_name
    abbreviated_name = words[0][0].upper() + '.'
    abbreviated_name += ' ' + ' '.join(words[1:])
    return abbreviated_name

def inverse_weighting(values):
    inverse_weights = [1 / value for value in values]
    total_inverse_weight = sum(inverse_weights)
    normalized_weights = [weight / total_inverse_weight for weight in inverse_weights]
    return normalized_weights