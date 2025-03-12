import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ["OMP_THREAD_LIMIT"] = "1"

import time
import pickle
import pandas as pd
import numpy as np
from joblib import parallel_backend
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingRegressor, BaggingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor

from .zir import ZeroInflatedRegressor
from .zero_stratified_kfold import ZeroStratifiedKFold,  UpsampledZeroStratifiedKFold
from .log_grid_search import LogGridSearch

class tune:
    """
    A class for model training, hyperparameter tuning, and cross-validation.

    Attributes
    ----------
    X_train : pd.DataFrame
        The feature matrix used for training the models.
    y : pd.Series
        The target variable.
    model_config : dict
        Configuration dictionary containing model and training parameters.
    regions : str or None, optional
        Name of the feature column representing regions, used for stratification (default is None).

    Methods
    -------
    train(model, classifier=False, regressor=False, log="no"):
        Train and tune models based on the provided configuration.
    """

    def __init__(self, X_train, y, model_config, regions=None):
        """
        Initialize the `tune` object.

        Parameters
        ----------
        X_train : pd.DataFrame of shape (n_samples, n_features)
            Training features used for model fitting.
        y : pd.Series of shape (n_samples,) or (n_samples, n_outputs)
            Target values used for model fitting.
        model_config : dict
            Dictionary containing model configuration parameters such as:
                - seed: int, random seed for reproducibility
                - root : str, path to Abil root folder
                - path_out : str, where predictions are saved
                - target : str, file name of your target list
                - verbose : int, to set verbosity (0-3)
                - n_threads : int, number of threads to use
                - cv : int, number of cross-folds
                - ensemble_config : dict
                    Dictionary containing ensemble set up:
                        - classifier: bool
                            Whether to train a classification model.
                        - regressor: bool
                            Whether to train a regression model.
                        - m{n}: str, model name (ex. m1: "rf", m2: "xgb" etc.)
        regions : str or None, optional
            Column name for regions to be used in preprocessing and stratification.

        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> with open('/home/phyto/Abil/configuration/example_model_config.yml', 'r') as f:
        ...    model_config = load(f, Loader=Loader)
        >>> X, y = example_data(y_name =  "Coccolithus pelagicus",
        ...                            n_samples=500, n_features=5, noise=20, 
        ...                            random_state=model_config['seed'])
        >>> m = tune(X, y, model_config)

        Returns
        -------
        m: object
            The model used for training.
        
        """
        self.y = y.sample(frac=1, random_state=model_config['seed']) #shuffle
        print("length of y:")
        print(len(self.y))
        self.y = self.y.values.ravel()
        self.X_train = X_train.sample(frac=1, random_state=model_config['seed']) #shuffle
        self.model_config = model_config
        self.ensemble_config = model_config['ensemble_config']
        self.seed = model_config['seed']
        self.target = y.name
        self.target_no_space = self.target.replace(' ', '_')
        self.n_jobs = model_config['n_threads']
        self.verbose = model_config['verbose'] 
        self.regions = regions
        self.path_out = os.path.join(model_config['root'], model_config['path_out'], model_config['run_name'])

        # Check for valid regions
        if regions is not None:
            if regions not in X_train.columns:
                raise ValueError("Regions defined but not in X_train. Did you mean regions=None?")

        # Setup cross-validation strategy
        if model_config['stratify']:
            if model_config['upsample']:
                self.cv = UpsampledZeroStratifiedKFold(n_splits=model_config['cv'])
                print("upsampling = True")
            else:
                self.cv = ZeroStratifiedKFold(n_splits=model_config['cv'])
        else:
            self.cv = KFold(n_splits=model_config['cv'])
             
        self.bagging_estimators = model_config.get('knn_bagging_estimators', None)

        # Preprocessor for features
        predictors = model_config['predictors'].copy()
        numeric_features = X_train.columns.get_indexer(X_train[predictors].columns)
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        if regions:
            model_config['predictors'].remove(regions)
            categorical_features = [self.regions]
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            self.preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        else:
            self.preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features)
            ])
            

    
    def train(self, model, classifier=False, regressor=False, log="no"):

        """
        Trains a machine learning model using the specified configuration.

        Parameters
        ----------
        model : str
            The type of model to train. Supported options:
            - 'rf' : Random Forest
            - 'knn' : K-Nearest Neighbors
            - 'xgb' : XGBoost
            - 'gp' : Gaussian Process
        classifier : bool, default=False
            Whether to train a classification model.
        regressor : bool, default=False
            Whether to train a regression model.
        log : str, default="no"
            Log transformation option:
            - 'yes' : Apply log transformation to the target variable.
            - 'no' : No transformation.
            - 'both' : Train both with and without log transformation.
        
        Examples
        --------
        >>> m.train(model="rf", regressor=True)

        Returns
        -------
        None
        """

        if model =="xgb":
            clf_estimator = XGBClassifier(nthread=1, random_state=self.seed)
            reg_estimator = XGBRegressor(nthread=1, random_state=self.seed, 
                                         objective='reg:tweedie')
        elif model=="knn":
            if self.bagging_estimators ==None:
                raise ValueError("number of bagging estimators not defined")
            else:
                clf_estimator = BaggingClassifier(estimator=KNeighborsClassifier(), 
                                                  n_estimators=self.bagging_estimators, 
                                                  random_state=self.seed)
                reg_estimator = BaggingRegressor(estimator=KNeighborsRegressor(), 
                                                 n_estimators=self.bagging_estimators, 
                                                 random_state=self.seed)
        elif model=="rf":
            clf_estimator = RandomForestClassifier(random_state=self.seed, oob_score=True)
            reg_estimator = RandomForestRegressor(random_state=self.seed, oob_score=True)
        elif model=="gp":
            from sklearn.gaussian_process.kernels import RBF

            kernel = 1 * RBF(length_scale=2.0, length_scale_bounds=(1e-2, 1e2))

            clf_estimator = GaussianProcessClassifier(random_state=self.seed)
            reg_estimator =  GaussianProcessRegressor(random_state=self.seed, kernel=kernel,
                                                       normalize_y=True)
        else:
            raise ValueError("invalid model")

        if (self.ensemble_config['classifier'] == False) and (self.ensemble_config['regressor'] == False):
            raise ValueError("both classifier and regressor defined as False")

        if (self.ensemble_config['classifier'] == True) and (self.ensemble_config['regressor'] != True):        
            raise ValueError("classifiers are not supported")

        if self.ensemble_config['regressor'] == True:
            if self.ensemble_config['classifier'] == True:
                y = self.y[self.y > 0]
                X_train = self.X_train[self.y > 0].reset_index(drop=True)
                cv = ZeroStratifiedKFold(n_splits=self.model_config['cv'])
            else:
                y = self.y
                X_train = self.X_train
                cv = self.cv

            print("training regressor")

            reg_scoring = {
                'R2': 'r2',
                'MAE': 'neg_mean_absolute_error',
                'RMSE': 'neg_root_mean_squared_error'
            }

            user_reg_param_grid = self.model_config['param_grid'][model + '_param_grid']['reg_param_grid']

            # Add the prefix 'regressor__estimator__' to each key
            reg_param_grid = {
                f"regressor__estimator__{key}": value
                for key, value in user_reg_param_grid.items()
            }

            print(reg_param_grid)

            reg_sav_out_scores = os.path.join(self.path_out, "scoring/", model)
            reg_sav_out_model = os.path.join(self.path_out, "model/", model)


            try: #make new dir if needed
                os.makedirs(reg_sav_out_scores)
            except:
                None

            try: #make new dir if needed
                os.makedirs(reg_sav_out_model)
            except:
                None            
                
            reg_pipe = Pipeline(steps=[('preprocessor', self.preprocessor),
                        ('estimator', reg_estimator)])
            
            with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
                reg = LogGridSearch(reg_pipe, verbose = self.verbose, cv=cv, 
                                    param_grid=reg_param_grid, scoring='r2', regions=self.regions)
                reg_grid_search = reg.transformed_fit(X_train, y, log, self.model_config['predictors'].copy())

            m2 = reg_grid_search.best_estimator_


            with open(os.path.join(reg_sav_out_model, self.target_no_space) + '_reg.sav', 'wb') as f:
                pickle.dump(m2, f)

            print("exported model to: " + reg_sav_out_model + "/"  + self.target_no_space + '_reg.sav')

            with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
                reg_scores = cross_validate(m2, X_train, y, cv = cv, verbose = self.verbose, scoring=reg_scoring)

            with open(os.path.join(reg_sav_out_scores, self.target_no_space) + '_reg.sav', 'wb') as f:
                pickle.dump(reg_scores, f)

            print("exported scoring to: " + reg_sav_out_scores + "/" + self.target_no_space + '_reg.sav')

            if "RMSE" in reg_scoring:
                try:
                    print("reg rRMSE: " + str(int(round(np.mean(reg_scores['test_RMSE'])/np.mean(self.y), 2)*-100))+"%")
                except:
                    print("reg rRMSE is NA (!)")
            if "MAE" in reg_scoring:
                try:
                    print("reg rMAE: " + str(int(round(np.mean(reg_scores['test_MAE'])/np.mean(self.y), 2)*-100))+"%")
                except:
                    print("reg rMAE is NA (!)")
            if "R2" in reg_scoring:
                try:
                    print("reg R2: " + str(round(np.mean(reg_scores['test_R2']), 2)))
                except:
                    print("reg R2 is NA (!)")

        if (self.ensemble_config['classifier'] == True) and (self.ensemble_config['regressor'] == True):      
            
            print("training classifier")

            user_clf_param_grid = self.model_config['param_grid'][model + '_param_grid']['clf_param_grid']

            # Add the prefix 'regressor__estimator__' to each key
            clf_param_grid = {
                f"estimator__{key}": value
                for key, value in user_clf_param_grid.items()
            }
            
            clf_scoring = {'accuracy': 'balanced_accuracy'}

            clf_sav_out_scores = os.path.join(self.path_out, "scoring/", model)
            clf_sav_out_model = os.path.join(self.path_out, "model/", model)


            try: #make new dir if needed
                os.makedirs(clf_sav_out_scores)
            except:
                None

            try: #make new dir if needed
                os.makedirs(clf_sav_out_model)
            except:
                None

            clf_pipe = Pipeline(steps=[('preprocessor', self.preprocessor),
                      ('estimator', clf_estimator)])

            clf = GridSearchCV(
                estimator=clf_pipe,
                param_grid= clf_param_grid,
                scoring= 'balanced_accuracy',
                cv = self.cv,
                verbose = self.verbose
            )

            y_clf =  self.y.copy()
            print("length of y_clf:")
            print(len(y_clf))

            y_clf[y_clf > 0] = 1
            print(y_clf)
            with parallel_backend('multiprocessing', self.n_jobs):
                clf.fit(self.X_train, y_clf)

            m1 = clf.best_estimator_
            
            with open(os.path.join(clf_sav_out_model, self.target_no_space) + '_clf.sav', 'wb') as f:
                pickle.dump(m1, f)
            
            print("exported model to:" + clf_sav_out_model + "/" + self.target_no_space + '_clf.sav')

            clf_scores = cross_validate(m1, self.X_train, y_clf, cv=self.cv, verbose =self.verbose, scoring=clf_scoring)
            
            with open(os.path.join(clf_sav_out_scores, self.target_no_space) + '_clf.sav', 'wb') as f:
                pickle.dump(clf_scores, f)
            
            print("exported scoring to: " + clf_sav_out_scores + "/" + self.target_no_space + '_clf.sav')

            print(clf_scores['test_accuracy'])
            print("clf balanced accuracy " + str((round(np.mean(clf_scores['test_accuracy']), 2))))

            print("training zero-inflated regressor")

            zir = ZeroInflatedRegressor(
                classifier=m1,
                regressor=m2,
            )

            zir_sav_out_scores = os.path.join(self.path_out, "scoring/", model)
            zir_sav_out_model = os.path.join(self.path_out, "model/", model)


            try: #make new dir if needed
                os.makedirs(zir_sav_out_scores)
            except:
                None

            try: #make new dir if needed
                os.makedirs(zir_sav_out_model)
            except:
                None           

            zir.fit(self.X_train, self.y)

            with open(os.path.join(zir_sav_out_model, self.target_no_space) + '_zir.sav', 'wb') as f:
                pickle.dump(zir, f)
                
            print("exported model to: " + zir_sav_out_model + "/" + self.target_no_space + '_zir.sav')

            with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
                zir_scores = cross_validate(zir, self.X_train, self.y, cv=self.cv, verbose =self.verbose, scoring=reg_scoring)

            with open(os.path.join(zir_sav_out_scores, self.target_no_space) + '_zir.sav', 'wb') as f:
                pickle.dump(zir_scores, f)

            print("exported scoring to: " + zir_sav_out_scores + "/" + self.target_no_space + '_zir.sav')

            try:
                print("zir rRMSE: " + str(int(round(np.mean(zir_scores['test_RMSE'])/np.mean(self.y), 2)*-100))+"%")
            except:
                print("zir rRMSE is NA (!)")
            try:
                print("zir rMAE: " + str(int(round(np.mean(zir_scores['test_MAE'])/np.mean(self.y), 2)*-100))+"%")
            except:
                print("zir rMAE is NA (!)")
            try:
                print("zir R2: " + str(round(np.mean(zir_scores['test_R2']), 2)))
            except:
                print("zir R2 is NA (!)")
        st = time.time()
        et = time.time()
        elapsed_time = et-st

        print("execution time:", elapsed_time, "seconds")        