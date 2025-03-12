import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings

from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.model_selection import KFold, cross_validate
from joblib import Parallel, delayed, parallel_backend  

from .utils import inverse_weighting, find_optimal_threshold
import shutil

# Set the custom temporary folder for loky
temp_folder = os.path.join(".","tmp") 
os.environ["LOKY_TEMP_FOLDER"] = temp_folder
# Ensure the directory exists
os.makedirs(temp_folder, exist_ok=True)

from . import unified_tree_or_bag as pp
from .zir import ZeroInflatedRegressor
from .zero_stratified_kfold import ZeroStratifiedKFold,  UpsampledZeroStratifiedKFold

def load_model_and_scores(path_out, ensemble_config, n, target):
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


def export_prediction(ensemble_config, m, target, target_no_space, X_predict, X_train, y_train, cv, model_out, n_threads=8):
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
    X_predict : pd.DataFrame of shape (n_points, n_features)
        Features to predict on (e.g., environmental data), where n_points
        is the total 1-d size of the features to predict on 
        (ex. 31881600 for full 180x360x41x12 grid).
    model_out : str
        Path where the predictions should be saved.
    n_threads : int, optional, default=1
        The number of threads to use for parallel prediction.
    """

    if (ensemble_config["classifier"] ==False) and (ensemble_config["regressor"] == True):
        with parallel_backend("loky", n_jobs=n_threads):
            d = pp.process_data_with_model(
                m, X_predict=X_predict, X_train=X_train, y_train=y_train, cv=cv
            )["predict_stats"]

        d = d.to_xarray()
        d['target'] = target
        export_path = os.path.join(model_out, target_no_space + ".nc")
        try: #make new dir if needed
            os.makedirs(model_out)
        except:
            None
        d.to_netcdf(export_path) 
        print("finished exporting summary stats to: ",  export_path)
        
    elif (ensemble_config["classifier"] ==True) and (ensemble_config["regressor"] == True):

        with parallel_backend("loky", n_jobs=n_threads):
            d_both = pp.process_data_with_model(
                m, X_predict=X_predict, X_train=X_train, y_train=y_train>0, cv=cv
            )
            # Generate classifier and regressor stats
            d_clf = d_both['classifier_predict_stats']
            d_reg = d_both['regressor_predict_stats']



        columns = ["mean", "sd", "median", "ci95_LL", "ci95_UL"]
        d = pd.DataFrame(d_reg)
        y_clf = y_train.copy()
        y_clf[y_clf > 0] = 1
        optimal_threshold = find_optimal_threshold(m.classifier, X_train, y_clf)
        for col in columns:
            d[col] = np.where(d_clf[col] < optimal_threshold, 0, d_reg[col])

        d_clf = d_clf.to_xarray()
        d_reg = d_reg.to_xarray()
        d = d.to_xarray()
        d_clf['target'] = target
        d_reg['target'] = target
        d['target'] = target

        clf_export_path = os.path.join(model_out, "clf", target_no_space + ".nc")
        reg_export_path = os.path.join(model_out,"reg", target_no_space + ".nc")
        zir_export_path = os.path.join(model_out, target_no_space + ".nc")

        for dir_name in ["", "clf", "reg"]:
            try:
                os.makedirs(os.path.join(model_out, dir_name))
            except FileExistsError:
                pass

        d_clf.to_netcdf(clf_export_path) 
        print("finished exporting summary stats to: ",  clf_export_path)
        d_reg.to_netcdf(reg_export_path) 
        print("finished exporting summary stats to: ",  reg_export_path)
        d.to_netcdf(zir_export_path) 
        print("finished exporting summary stats to: ",  zir_export_path)

    else:
        raise ValueError("classifiers are not supported")

    #remove loky tmp data:
    shutil.rmtree(temp_folder, ignore_errors=True)

class predict:
    """
    Predict outcomes using an ensemble of regression models and export the predictions to a NetCDF file.

    Parameters
    ----------
    X_train : pd.DataFrame of shape (n_samples, n_features)
        Training features used for model fitting.
    y : pd.Series of shape (n_samples,) or (n_samples, n_outputs)
        Target values used for model fitting.
    X_predict : pd.DataFrame of shape (n_points, n_features)
        Features to predict on, where `n_points` represents the total number of prediction points 
        (e.g., 31881600 for a full 180x360x41x12 grid).
    model_config : dict
        Dictionary containing model configuration parameters, including:
            - seed : int
                Random seed for reproducibility.
            - root : str
                Path to the root folder.
            - path_out : str
                Directory where predictions are saved.
            - target : str
                File name of the target list.
            - verbose : int
                Verbosity level (0-3).
            - n_threads : int
                Number of threads to use for parallel processing.
            - cv : int
                Number of cross-validation folds.
            - ensemble_config : dict
                Configuration for the ensemble setup, containing:
                    - classifier : bool
                        Whether to train a classification model.
                    - regressor : bool
                        Whether to train a regression model.
                    - m{n} : str
                        Model names (e.g., "m1: 'rf'", "m2: 'xgb'").
    n_jobs : int, optional, default=1
        Number of threads to use for parallel processing.

    Attributes
    ----------
    path_out : str
        Path where predictions and model outputs are saved.
    target : str
        Name of the target variable.
    target_no_space : str
        Target variable name with spaces replaced by underscores.
    verbose : int
        Verbosity level for logging.
    n_jobs : int
        Number of parallel threads used for prediction and cross-validation.

    Methods
    -------
    make_prediction()
        Train the ensemble models and generate predictions, exporting them to NetCDF.
    """

    def __init__(self, X_train, y, X_predict, model_config, n_jobs=1):
        """
        Initialize the `predict` class with training data, prediction data, and model configurations.

        Parameters
        ----------
        X_train : pd.DataFrame of shape (n_samples, n_features)
            Training features used for model fitting.
        y : pd.Series of shape (n_samples,) or (n_samples, n_outputs)
            Target values used for model fitting.
        X_predict : pd.DataFrame of shape (n_points, n_features)
            Features for which predictions are to be made.
        model_config : dict
            Dictionary containing configuration parameters for the model and ensemble.
        n_jobs : int, optional, default=1
            Number of threads for parallel processing.

        Returns
        -------
        None
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
            self.scoring = {
                'R2': 'r2',
                'MAE': 'neg_mean_absolute_error',
                'RMSE': 'neg_root_mean_squared_error'
            }


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
        Fit models in the ensemble and generate predictions.

        Predictions are exported to NetCDF files. If the ensemble contains multiple models, 
        predictions are made for each individual model and the ensemble.

        Returns
        -------
        None

        Notes
        -----
        - Individual model predictions and ensemble predictions are saved separately.
        - Performance metrics (e.g., cross-validation scores) are saved for the ensemble.
        - Only regression models are supported; classification is not implemented.
        """

        number_of_models = len(self.ensemble_config) -2
        print("number of models in ensemble:" + str(number_of_models))

        if number_of_models==1:

            m, mae1 = load_model_and_scores(self.path_out, self.ensemble_config, 0, self.target_no_space)

            model_name = self.ensemble_config["m" + str(1)]
            model_out = os.path.join(self.path_out, "predictions", model_name)

            # export_prediction(m=m, target = self.target, target_no_space = self.target_no_space, X_predict = self.X_predict,
            #                   model_out = model_out, n_threads=self.n_jobs)
            export_prediction(self.ensemble_config, m, self.target, self.target_no_space, self.X_predict, self.X_train, self.y, self.cv, 
                              model_out, n_threads=self.n_jobs)

        elif number_of_models >=2:
                    
            # iteratively make prediction for each model
            models = []
            mae_values = []
            w = []

            for i in range(number_of_models):
                m, mae = load_model_and_scores(self.path_out, self.ensemble_config, i, self.target_no_space)
                model_name = self.ensemble_config["m" + str(i + 1)]
                model_out = os.path.join(self.path_out, "predictions", model_name)
                
                print("===================")
                print("DEBUG OF ZIR MODELS")
                print("===================")
                print(model_name)
                print(type(m))
                print(type(self.X_predict))
                print(type(self.X_train))


                if (self.ensemble_config["classifier"] ==False) and (self.ensemble_config["regressor"] == True):
                    export_prediction(self.ensemble_config, m, self.target, self.target_no_space, self.X_predict, self.X_train, self.y, self.cv, 
                                    model_out, n_threads=self.n_jobs)
                    print("exporting " + model_name + " prediction to: " + model_out)
                if (self.ensemble_config["classifier"] ==True) and (self.ensemble_config["regressor"] == True):

                    export_prediction(self.ensemble_config, m, self.target, self.target_no_space, self.X_predict, self.X_train, self.y, self.cv, 
                                     model_out, n_threads=self.n_jobs)

                models.append((model_name, m))
                mae_values.append(mae)

            w = inverse_weighting(mae_values) 
            if (self.ensemble_config["classifier"] ==False) and (self.ensemble_config["regressor"] == True):
                m = VotingRegressor(estimators=models, weights=w).fit(self.X_train, self.y)   
                model_out = os.path.join(self.path_out, "predictions", "ens")
                # export_prediction(m=m, target = self.target, target_no_space = self.target_no_space, X_predict = self.X_predict,
                #               model_out = model_out, n_threads=self.n_jobs)      
                export_prediction(self.ensemble_config, m, self.target, self.target_no_space, self.X_predict, self.X_train, self.y, self.cv, 
                                model_out, n_threads=self.n_jobs)
                
                #export model object:
                base_output_path = os.path.join(self.path_out, "model", "ens")
                try: #make new dir if needed
                    os.makedirs(base_output_path)
                except:
                    None

                file_path = os.path.join(base_output_path, f"{self.target_no_space}{self.extension}")

                with open(file_path, 'wb') as f:
                    pickle.dump(m, f)

                print("exporting m ens to: ", file_path)
            elif (self.ensemble_config["classifier"] ==True) and (self.ensemble_config["regressor"] == True):
                clf_models = []
                reg_models = []

                for i in range(number_of_models):
                    path_to_param  = os.path.join(self.path_out, "model", self.ensemble_config["m"+str(i+1)])
                    print("loading clf")

                    with open(os.path.join(path_to_param, self.target_no_space) + '_clf.sav', 'rb') as file:
                        m_clf = pickle.load(file)       
                    clf_models.append(('clf'+str(i+1), m_clf))  

                    print("loading reg")
                    with open(os.path.join(path_to_param, self.target_no_space) + '_reg.sav', 'rb') as file:
                        m_reg = pickle.load(file)

                    reg_models.append(('reg'+str(i+1), m_reg))  

                print("defining voting regressor")
                y = self.y.values.ravel().copy()

                voting_reg = VotingRegressor(estimators=reg_models, weights=w).fit(self.X_train, y)
                y_clf = y.copy()
                y_clf[y_clf > 0] = 1

                voting_clf = VotingClassifier(estimators=clf_models, weights=w, voting="soft").fit(self.X_train, y_clf)

                optimal_threshold = find_optimal_threshold(voting_clf, self.X_train, y_clf)
                print("optimal clf threshold: ", optimal_threshold)
                print("defining ZIR")
                m = ZeroInflatedRegressor(
                    classifier=voting_clf,
                    regressor=voting_reg,
                    threshold=optimal_threshold
                )
                print("predicting ZIR")
                m.fit(self.X_train, y)

                model_out = os.path.join(self.path_out, "predictions", "ens")

                # export_prediction(m=m, target = self.target, target_no_space = self.target_no_space, X_predict = self.X_predict,
                #               model_out = model_out, n_threads=self.n_jobs)  
                export_prediction(self.ensemble_config, m, self.target, self.target_no_space, self.X_predict, self.X_train, self.y, self.cv, 
                                model_out, n_threads=self.n_jobs)
                print("exporting ZIR object")
                base_output_path = os.path.join(self.path_out, "model", "ens")

                try: #make new dir if needed
                    os.makedirs(base_output_path)
                except:
                    None

                file_path = os.path.join(base_output_path, f"{self.target_no_space}{self.extension}")

                with open(file_path, 'wb') as f:
                    pickle.dump(m, f)

                print("exporting to ens: ", file_path)

            else:
                raise ValueError("classifiers are not supported")

            scores = cross_validate(m, self.X_train, self.y, cv=self.cv, verbose=self.verbose, 
                                    scoring=self.scoring, n_jobs=self.n_jobs)

            model_out_scores = os.path.join(self.path_out, "scoring", "ens")

            try: #make new dir if needed
                os.makedirs(model_out_scores)
            except:
                None

            scores_file_path = os.path.join(model_out_scores, f"{self.target_no_space}{self.extension}")


            with open(scores_file_path, 'wb') as f:
                pickle.dump(scores, f)
            print("exporting ensemble scores to: " + scores_file_path)

        else:
            raise ValueError("at least one model should be defined in the ensemble")

        et = time.time()
        elapsed_time = et-self.st
        print("finished")
        print("execution time:", elapsed_time, "seconds")
