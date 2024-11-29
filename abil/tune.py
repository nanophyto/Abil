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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingRegressor, BaggingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor


if 'site-packages' in __file__ or os.getenv('TESTING') == 'true':
    from abil.functions import ZeroInflatedRegressor, LogGridSearch, ZeroStratifiedKFold, UpsampledZeroStratifiedKFold
else:
    from functions import  ZeroInflatedRegressor, LogGridSearch, ZeroStratifiedKFold, UpsampledZeroStratifiedKFold

class tune:
    """
    Parameters
    ----------

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The training input samples. Internally, its dtype will be converted
        to ``dtype=np.float32``. If a sparse matrix is provided, it will be
        converted into a sparse ``csc_matrix``.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in
        regression).

    model_config: dictionary, default=None
        A dictionary containing:

        `seed` : int, used to create random numbers
        
        `root`: string, path to folder
        
        `path_out`: string, where predictions are saved
        
        `path_in`: string, where to find tuned models
        
        `traits`: string, file name of your trait file
        
        `verbose`: int, to set verbosity (0-3)
        
        `n_threads`: int, number of threads to use
        
        `cv` : int, number of cross-folds
                    
        `ensemble_config` : 
        
        `clf_scoring` :
        
        `reg_scoring` :    
    
    """
    def __init__(self, X_train, y, model_config, regions=None):

        """
        simulate-pseudo-absence 
        if True:
        1) run two phase model.
        2) Drop zeros for regression
        3) Include zeros for 2-phase fitting + validation

        if False, None
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

        if regions is not None:
            if regions not in X_train.columns:
                raise ValueError("Regions defined but not in X_train. Did you mean regions=None?")

        if model_config['stratify']==True:
            if model_config['upsample']==True:
                self.cv = UpsampledZeroStratifiedKFold(n_splits=model_config['cv'])
                print("upsampling = True")
            else:
                self.cv = ZeroStratifiedKFold(n_splits=model_config['cv'])
        else:
            self.cv = KFold(n_splits=model_config['cv'])
             
        try:
            self.bagging_estimators = model_config['knn_bagging_estimators'] 
        except:
            self.bagging_estimators = None


        if regions!=None:
            model_config['predictors'].remove(regions)
            categorical_features = [self.regions]
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        predictors = model_config['predictors'].copy()

        numeric_features =  self.X_train.columns.get_indexer(self.X_train[predictors].columns)
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])

        if self.regions!=None:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)])
            
        else:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)])
            

    
    def train(self, model, classifier=False, regressor=False, log="no"):

        """

        Parameters
        ----------
        model : string, default="rf"
            Which model to train: 
            Supported models:
            `"rf"` Random Forest 
            `"knn"` K-Nearest Neighbors
            `"xgb"` XGBoost

        classifier : bool, default=False

        regressor : bool, default=False

        log : string, default="no"
            If `"yes"`, log transformation is applied to y
            
            If `"no"`, y is not transformed
            
            If `"both"`, both log and no-log transformations are fitted by 
                running the model two times.



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
        >>> m.train(model="rf", regressor=True)

        
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

            reg_scoring = self.model_config['reg_scoring']

            reg_param_grid = self.model_config['param_grid'][model + '_param_grid']['reg_param_grid']

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
                print("reg rRMSE: " + str(int(round(np.mean(reg_scores['test_RMSE'])/np.mean(self.y), 2)*-100))+"%")
            if "MAE" in reg_scoring:
                print("reg rMAE: " + str(int(round(np.mean(reg_scores['test_MAE'])/np.mean(self.y), 2)*-100))+"%")
            if "R2" in reg_scoring:
                print("reg R2: " + str(round(np.mean(reg_scores['test_R2']), 2)))


        if (self.ensemble_config['classifier'] == True) and (self.ensemble_config['regressor'] == True):      
            
            print("training classifier")
            clf_param_grid = self.model_config['param_grid'][model + '_param_grid']['clf_param_grid']
            clf_scoring = self.model_config['clf_scoring']

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

            print("zir rRMSE: " + str(int(round(np.mean(zir_scores['test_RMSE'])/np.mean(self.y), 2)*-100))+"%")
            print("zir rMAE: " + str(int(round(np.mean(zir_scores['test_MAE'])/np.mean(self.y), 2)*-100))+"%")
            print("zir R2: " + str(round(np.mean(zir_scores['test_R2']), 2)))

        st = time.time()
        et = time.time()
        elapsed_time = et-st

        print("execution time:", elapsed_time, "seconds")        

    """

    """