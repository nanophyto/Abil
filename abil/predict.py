

import pandas as pd
import numpy as np
import pickle
import os
import time

from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import KFold, cross_validate
from joblib import Parallel, delayed


if 'site-packages' in __file__ or os.getenv('TESTING') == 'true':
    from abil.functions import inverse_weighting, ZeroStratifiedKFold,  UpsampledZeroStratifiedKFold
else:
    from functions import inverse_weighting, ZeroStratifiedKFold,  UpsampledZeroStratifiedKFold

def def_prediction(path_out, ensemble_config, n, target):
    """
    Loads a trained model and scoring information, and calculates the mean absolute error (MAE) for the prediction.

    Parameters
    ----------
    path_out : str
        Path to the output folder containing model and scoring information.
    ensemble_config : dict
        Dictionary containing configuration details for the ensemble models, including model names, regressor or classifier status.
    n : int
        Index of the model to load in the ensemble.
    target : str
        The target for which predictions are made (used to load target-specific files).

    Returns
    -------
    tuple
        A tuple containing the model and the mean absolute error (MAE) score.

    Raises
    ------
    ValueError
        If both regressor and classifier are set to False, or if a classifier is used when only regressors are supported.
    """

    path_to_scores  = os.path.join(path_out, "scoring", ensemble_config["m"+str(n+1)])
    path_to_param  = os.path.join(path_out, "model", ensemble_config["m"+str(n+1)])

    if (ensemble_config["classifier"] ==True) and (ensemble_config["regressor"] == False):
        raise ValueError("classifiers are not supported")

    elif (ensemble_config["classifier"] ==False) and (ensemble_config["regressor"] == True):
        print("predicting regressor")
        target_no_space = target.replace(' ', '_')
        with open(os.path.join(path_to_param, target_no_space) + '_reg.sav', 'rb') as file:
            m = pickle.load(file)
        with open(os.path.join(path_to_scores, target_no_space) + '_reg.sav', 'rb') as file:
            scoring = pickle.load(file) 
        scores = abs(np.mean(scoring['test_MAE']))


    elif (ensemble_config["classifier"] ==True) and (ensemble_config["regressor"] == True):
        print("predicting zero-inflated regressor")
        target_no_space = target.replace(' ', '_')
        with open(os.path.join(path_to_param, target_no_space) + '_zir.sav', 'rb') as file:
            m = pickle.load(file)
        with open(os.path.join(path_to_scores, target_no_space) + '_zir.sav', 'rb') as file:
            scoring = pickle.load(file)
        scores = abs(np.mean(scoring['test_MAE']))

    elif (ensemble_config["classifier"] ==False) and (ensemble_config["regressor"] == False):
        raise ValueError("Both regressor and classifier are defined as false")

    return(m, scores)


def parallel_predict(prediction_function, X_predict, n_threads=1):
    """
    Splits the prediction task across multiple threads to predict on large datasets.

    Parameters
    ----------
    prediction_function : callable
        The model's prediction function to be applied to each chunk of data.
    X_predict : DataFrame
        The features (input data) on which to make predictions.
    n_threads : int, optional, default=1
        The number of threads to use for parallel processing.

    Returns
    -------
    np.ndarray
        The combined predictions from all threads.
    """    

    # Split the indices of X_predict into chunks
    chunk_indices = np.array_split(X_predict.index, n_threads)


    # Create a list of DataFrame chunks based on the split indices
    df_sections = [X_predict.loc[chunk_idx] for chunk_idx in chunk_indices]

    # Use joblib to process each chunk in parallel
    predictions = Parallel(n_jobs=n_threads)(
        delayed(prediction_function)(df_section) for df_section in df_sections
    )

    # Combine the predictions from all threads
    combined_predictions = np.concatenate(predictions)

    return combined_predictions


def export_prediction(m, target, target_no_space, X_predict, model_out, n_threads=1):
    """
    Exports model predictions to a NetCDF file.

    Parameters
    ----------
    m : object
        The trained model used for predictions.
    target : str
        The name of the target variable.
    target_no_space : str
        The target variable name with spaces replaced by underscores.
    X_predict : DataFrame
        The features on which to make predictions.
    model_out : str
        Path where the predictions should be saved.
    n_threads : int, optional, default=1
        The number of threads to use for parallel prediction.
    """

    d = X_predict.copy()
    d[target] = parallel_predict(m.predict, X_predict, n_threads)
    d = d.to_xarray()
    
    try: #make new dir if needed
        os.makedirs(model_out)
    except:
        None

    export_path = os.path.join(model_out, target_no_space + ".nc")


    d[target].to_netcdf(export_path) 


class predict:
    """
    Predicts outcomes based on an ensemble of regression models and exports the predictions to a NetCDF file.

    Parameters
    ----------
    X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training features used for model fitting.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target values used for model fitting.
    X_predict : {array-like, sparse matrix} of shape (n_samples, n_features)
        Features to predict on (e.g., environmental data).
    model_config : dict
        Dictionary containing model configuration parameters such as:
        - seed: int, random seed for reproducibility
        - path_out: str, output path for saving results
        - path_in: str, input path to models
        - verbose: int, verbosity level (0-3)
        - cv: int, number of cross-validation folds
        - ensemble_config: dict, configuration for ensemble models
    n_jobs : int, optional, default=1
        The number of threads to use for parallel processing.

    Attributes
    ----------
    path_out : str
        The path where predictions and model outputs are saved.
    target : str
        The name of the target variable.
    target_no_space : str
        The target variable name with spaces replaced by underscores.
    verbose : int
        Verbosity level for the model.
    n_jobs : int
        The number of parallel threads for prediction and cross-validation.

    Methods
    -------
    make_prediction()
        Fits the ensemble model(s) and makes predictions, exporting them to NetCDF.
    """

    def __init__(self, X_train, y, X_predict, model_config, n_jobs=1):
        """
        Initializes the prediction process by setting up model configurations, cross-validation, and paths.
        """
                
        self.st = time.time()

        self.y = y.sample(frac=1, random_state=model_config['seed']) #shuffle
        self.X_train = X_train.sample(frac=1, random_state=model_config['seed']) #shuffle

        self.seed = model_config['seed']
        self.target = y.name
        self.target_no_space = self.target.replace(' ', '_')
        self.verbose = model_config['verbose']

        self.path_out = os.path.join(model_config['root'], model_config['path_out'], model_config['run_name'])

            
        if model_config['stratify']==True:
            if model_config['upsample']==True:
                self.cv = UpsampledZeroStratifiedKFold(n_splits=model_config['cv'])
                print("upsampling = True")
            else:
                self.cv = ZeroStratifiedKFold(n_splits=model_config['cv'])
        else:
            self.cv = KFold(n_splits=model_config['cv'])

        self.X_predict = X_predict
        X_predict = None
        self.ensemble_config = model_config['ensemble_config']
        self.model_config = model_config

        self.n_jobs = n_jobs

        if (self.ensemble_config["classifier"] ==True) and (self.ensemble_config["regressor"] == False):
            raise ValueError("classifiers are not supported")
        elif (self.ensemble_config["classifier"] ==False) and (self.ensemble_config["regressor"] == False):
            raise ValueError("classifier and regressor can't both be False")
        else:
            self.scoring = self.model_config['reg_scoring']

        if (self.ensemble_config["regressor"] !=True) and (self.ensemble_config["regressor"] !=False):
            raise ValueError("regressor should be True or False")


        if self.model_config['ensemble_config']['classifier'] and not self.model_config['ensemble_config']['regressor']:
            raise ValueError("classifiers are not supported")
        elif self.model_config['ensemble_config']['classifier'] and self.model_config['ensemble_config']['regressor']:
            self.extension = "_zir.sav"
        else:
            self.extension = "_reg.sav"

        print("initialized prediction")
        
    def make_prediction(self):

        """
        Fits the models in the ensemble and makes predictions. Exports the predictions and model performance metrics.

        If multiple models are provided, predictions are made for both individual models and an ensemble of models.
        
        Returns
        -------
        None

        Notes
        -----
        If more than one model is provided, predictions are made for both 
        individual models and an ensemble of the models. 

        """

        number_of_models = len(self.ensemble_config) -2
        print("number of models in ensemble:" + str(number_of_models))

        if number_of_models==1:

            m, mae1 = def_prediction(self.path_out, self.ensemble_config, 0, self.target_no_space)

            model_name = self.ensemble_config["m" + str(1)]
            model_out = os.path.join(self.path_out, "predictions", model_name)

            export_prediction(m=m, target = self.target, target_no_space = self.target_no_space, X_predict = self.X_predict,
                              model_out = model_out, n_threads=self.n_jobs)

        elif number_of_models >=2:
                    
            # iteratively make prediction for each model
            models = []
            mae_values = []
            w = []

            for i in range(number_of_models):
                m, mae = def_prediction(self.path_out, self.ensemble_config, i, self.target_no_space)
                model_name = self.ensemble_config["m" + str(i + 1)]
                model_out = os.path.join(self.path_out, "predictions", model_name, "50")

                export_prediction(m=m, target = self.target, target_no_space = self.target_no_space, X_predict = self.X_predict,
                              model_out = model_out, n_threads=self.n_jobs)

                print("exporting " + model_name + " prediction to: " + model_out)

                models.append((model_name, m))
                mae_values.append(mae)

            w = inverse_weighting(mae_values) 

            if self.ensemble_config["regressor"] ==True:
                m = VotingRegressor(estimators=models, weights=w).fit(self.X_train, self.y)   
                model_out = os.path.join(self.path_out, "predictions", "ens", "50")
                export_prediction(m=m, target = self.target, target_no_space = self.target_no_space, X_predict = self.X_predict,
                              model_out = model_out, n_threads=self.n_jobs)                
            else:
                raise ValueError("classifiers are not supported")

            print(np.min(self.y))

            scores = cross_validate(m, self.X_train, self.y, cv=self.cv, verbose=self.verbose, 
                                    scoring=self.scoring, n_jobs=self.n_jobs)

            model_out_scores = os.path.join(self.path_out, "scoring", "ens")

            try: #make new dir if needed
                os.makedirs(model_out_scores)
            except:
                None

            with open(os.path.join(model_out_scores, self.target_no_space) + self.extension, 'wb') as f:
                pickle.dump(scores, f)
            print("exporting ensemble scores to: " + model_out_scores + self.target_no_space + self.extension)

        else:
            raise ValueError("at least one model should be defined in the ensemble")

        et = time.time()
        elapsed_time = et-self.st
        print("finished")
        print("execution time:", elapsed_time, "seconds")