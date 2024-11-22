import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate
from joblib import Parallel, delayed


if 'site-packages' in __file__ or os.getenv('TESTING') == 'true':
    from abil.functions import inverse_weighting, ZeroStratifiedKFold,  UpsampledZeroStratifiedKFold
else:
    from functions import inverse_weighting, ZeroStratifiedKFold,  UpsampledZeroStratifiedKFold

def def_prediction(path_out, ensemble_config, n, species):

    path_to_scores  = path_out + "scoring/" + ensemble_config["m"+str(n+1)] + "/"
    path_to_param  = path_out + "model/"  +  ensemble_config["m"+str(n+1)] + "/"

    if (ensemble_config["classifier"] ==True) and (ensemble_config["regressor"] == False):
        raise ValueError("classifiers are not supported")

    elif (ensemble_config["classifier"] ==False) and (ensemble_config["regressor"] == True):
        print("predicting regressor")
        species_no_space = species.replace(' ', '_')
        with open(path_to_param + species_no_space + '_reg.sav', 'rb') as file:
            m = pickle.load(file)
        with open(path_to_scores + species_no_space + '_reg.sav', 'rb') as file:
            scoring = pickle.load(file) 
        scores = abs(np.mean(scoring['test_MAE']))


    elif (ensemble_config["classifier"] ==True) and (ensemble_config["regressor"] == True):
        print("predicting zero-inflated regressor")
        species_no_space = species.replace(' ', '_')
        with open(path_to_param + species_no_space + '_zir.sav', 'rb') as file:
            m = pickle.load(file)
        with open(path_to_scores + species_no_space + '_zir.sav', 'rb') as file:
            scoring = pickle.load(file)
        scores = abs(np.mean(scoring['test_MAE']))

    elif (ensemble_config["classifier"] ==False) and (ensemble_config["regressor"] == False):
        raise ValueError("Both regressor and classifier are defined as false")

    return(m, scores)


def parallel_predict(prediction_function, X_predict, n_threads=1):

    df_sections = np.array_split(X_predict,  n_threads)

    predictions = Parallel(n_jobs= n_threads)(delayed(prediction_function)(df_section) for df_section in df_sections)

    combined_predictions = np.concatenate(predictions)

    return(combined_predictions)


def export_prediction(m, target, target_no_space, X_predict, model_out, n_threads=1):

    d = X_predict.copy()
    d[target] = parallel_predict(m.predict, X_predict, n_threads)
    d = d.to_xarray()
    
    try: #make new dir if needed
        os.makedirs(model_out)
    except:
        None

    d[target].to_netcdf(model_out + target_no_space + ".nc") 


class predict:
    """
    Parameters
    ----------

    X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
        should be the same as the X used for tuning

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        should be the same as the y used for tuning

    X_predict : {array-like, sparse matrix} of shape (n_samples, n_features)
        Features to predict on (i.e. gridded environmental data). 
            
    model_config: dictionary, default=None
        A dictionary containing:

        `seed` : int, used to create random numbers

        `root`: string, path to folder
        
        `path_out`: string, where predictions are saved
        
        `path_in`: string, where to find tuned models
        
        `traits`: string, file name of your trait file
        
        `verbose`: int, to set verbosity (0-3)
                
        `cv` : int, number of cross-folds

        `n_threads`: int, number of threads to use
                
        `ensemble_config` : 
        
        `clf_scoring` :
        
        `reg_scoring` :

    """
    def __init__(self, X_train, y, X_predict, model_config, n_jobs=1):
        
        self.st = time.time()

        self.y = y.sample(frac=1, random_state=model_config['seed']) #shuffle
        self.X_train = X_train.sample(frac=1, random_state=model_config['seed']) #shuffle

        self.seed = model_config['seed']
        self.target = y.name
        self.target_no_space = self.target.replace(' ', '_')
        self.verbose = model_config['verbose']

        if model_config['hpc']==False:
            self.path_out = model_config['local_root'] + model_config['path_out'] + model_config['run_name'] + "/"
        elif model_config['hpc']==True:
            self.path_out = model_config['hpc_root'] + model_config['path_out'] + model_config['run_name'] + "/"
        else:
            raise ValueError("hpc True or False not defined in yml")
            
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

        #unsure if this is required...
#        if (self.ensemble_config["classifier"] !=True) and (self.ensemble_config["classifier"] !=False):
#            raise ValueError("classifier should be True or False")
        
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
        Calculates performance of model(s) and exports prediction(s) to netcdf

        If more than one model is defined an weighted ensemble is generated using 
        a voting Regressor.
        
        Notes
        -----
        If more than one model is provided, predictions are made for both 
        invidiual models and an ensemble of the models. 

        """

        number_of_models = len(self.ensemble_config) -2
        print("number of models in ensemble:" + str(number_of_models))

        if number_of_models==1:

            m, mae1 = def_prediction(self.path_out, self.ensemble_config, 0, self.target_no_space)

            model_name = self.ensemble_config["m" + str(1)]
            model_out = self.path_out + "predictions/" + model_name + "/" 
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
                model_out = self.path_out + "predictions/ens/50/" #temporary until tree/bag CI is implemented! 
                export_prediction(m=m, target = self.target, target_no_space = self.target_no_space, X_predict = self.X_predict,
                              model_out = model_out, n_threads=self.n_jobs)

                print("exporting " + model_name + " prediction to: " + model_out)

                models.append((model_name, m))
                mae_values.append(mae)

            w = inverse_weighting(mae_values) 

            if self.ensemble_config["regressor"] ==True:
                m = VotingRegressor(estimators=models, weights=w).fit(self.X_train, self.y)   
            else:
                raise ValueError("classifiers are not supported")

            print(np.min(self.y))

            scores = cross_validate(m, self.X_train, self.y, cv=self.cv, verbose=self.verbose, 
                                    scoring=self.scoring, n_jobs=self.n_jobs)

            model_out_scores = self.path_out + "scoring/ens/"
            try: #make new dir if needed
                os.makedirs(model_out_scores)
            except:
                None

            with open(model_out_scores + self.target_no_space + self.extension, 'wb') as f:
                pickle.dump(scores, f)
            print("exporting ensemble scores to: " + model_out_scores + self.target_no_space + self.extension)

        else:
            raise ValueError("at least one model should be defined in the ensemble")

        et = time.time()
        elapsed_time = et-self.st
        print("finished")
        print("execution time:", elapsed_time, "seconds")