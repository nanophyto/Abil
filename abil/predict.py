import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.model_selection import KFold
from mapie.regression import MapieRegressor
from mapie.classification import  MapieClassifier
from mapie.conformity_scores import GammaConformityScore, AbsoluteConformityScore


from sklearn.model_selection import cross_validate
from joblib import Parallel, delayed


if 'site-packages' in __file__ or os.getenv('TESTING') == 'true':
    from abil.functions import inverse_weighting, ZeroStratifiedKFold,  UpsampledZeroStratifiedKFold, check_tau
else:
    from functions import inverse_weighting, ZeroStratifiedKFold,  UpsampledZeroStratifiedKFold, check_tau

def def_prediction(path_out, ensemble_config, n, species):

    path_to_scores  = path_out + "scoring/" + ensemble_config["m"+str(n+1)] + "/"
    path_to_param  = path_out + "model/"  +  ensemble_config["m"+str(n+1)] + "/"


    if (ensemble_config["classifier"] ==True) and (ensemble_config["regressor"] == False):
        print("predicting classifier")
        species_no_space = species.replace(' ', '_')
        with open(path_to_param + species_no_space + '_clf.sav', 'rb') as file:
            m = pickle.load(file)
        with open(path_to_scores + species_no_space + '_clf.sav', 'rb') as file:
            scoring = pickle.load(file)
        scores = np.mean(scoring['test_accuracy'])

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

        print("Both regressor and classifier are defined as false")

    return(m, scores)


def parallel_predict(prediction_function, X_predict, n_threads=1):

    df_sections = np.array_split(X_predict,  n_threads)

    predictions = Parallel(n_jobs= n_threads)(delayed(prediction_function)(df_section) for df_section in df_sections)

    combined_predictions = np.concatenate(predictions)

    return(combined_predictions)


def parallel_predict_mapie(X_predict, mapie, alpha, chunksize = 1000):

    # Define a function to make predictions and PIs for a chunk of data
    def predict_chunk(model, X_chunk, alpha):
        pred, pis = model.predict(X_chunk, alpha=alpha)
        return pred, pis

    # Define the chunk size

    # Split the data into chunks
    data_chunks = [X_predict[i:i+chunksize] for i in range(0, len(X_predict), chunksize)]

    # Use parallel processing to make predictions and PIs for each chunk
    results = Parallel(n_jobs=-1)(delayed(predict_chunk)(mapie, chunk, alpha) for chunk in data_chunks)

    # Combine the results
    y_pred = []
    y_pis = []
    for pred, pis in results:
        y_pred.extend(pred)
        y_pis.extend(pis)

    return np.array(y_pred), np.array(y_pis)
    # Now y_pred contains the predicted values and y_pis contains the prediction intervals



def export_prediction(m, species, X_predict, model_config, 
                      ensemble_config, ens_model_out, n_threads=1):

    d = X_predict.copy()
    if (model_config['predict_probability'] == True) and (ensemble_config["regressor"] ==False):
        print("predicting probabilities")
        d[species] = parallel_predict(m.predict_proba, X_predict, n_threads)
    elif (model_config['predict_probability'] == True) and (ensemble_config["regressor"] ==True):
        print("error: can't predict probabilities if the model is a regressor")
    else:
        d[species] = parallel_predict(m.predict, X_predict, n_threads)
    d = d.to_xarray()
    
    try: #make new dir if needed
        os.makedirs(ens_model_out)
    except:
        None

    d[species].to_netcdf(ens_model_out + species + ".nc") 



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
            self.scoring = model_config['clf_scoring']
            self.y[self.y > 0] = 1

        elif (self.ensemble_config["classifier"] ==False) and (self.ensemble_config["regressor"] == False):
            raise ValueError("classifier and regressor can't both be False")
        else:
            self.scoring = check_tau(self.model_config['reg_scoring']) 


        if (self.ensemble_config["classifier"] !=True) and (self.ensemble_config["classifier"] !=False):
            raise ValueError("classifier should be True or False")
        
        if (self.ensemble_config["regressor"] !=True) and (self.ensemble_config["regressor"] !=False):
            raise ValueError("regressor should be True or False")


        if self.model_config['ensemble_config']['classifier'] and not self.model_config['ensemble_config']['regressor']:
            self.extension = "_clf.sav"
        elif self.model_config['ensemble_config']['classifier'] and self.model_config['ensemble_config']['regressor']:
            self.extension = "_zir.sav"
        else:
            self.extension = "_reg.sav"

        print("initialized prediction")
        
    def make_prediction(self, prediction_inference=False, alpha=[0.05], cv=None,
                        conformity_score = AbsoluteConformityScore()):

        """
        Calculates performance of model(s) and exports prediction(s) to netcdf

        If more than one model is defined an weighted ensemble is generated using 
        a voting Regressor or Classifier.

        To determine model error Prediction Inference is implemented using MAPIE.

        Parameters
        ----------
        prediction_inference: Optional[bool]
            Whether or not to include prediction inference.
            
        alpha: Optional[float]
            Must be float between ``0`` and ``1``, represents the uncertainty of the
            confidence interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

            By default ``[0.05]`.

        conformity_score: Optional[ConformityScore]
            ConformityScore instance.
            It defines the link between the observed values, the predicted ones
            and the conformity scores. 

            - ConformityScore: any MAPIE ``ConformityScore`` class

            By default ``AbsoluteConformityScore()``.


        cv: Optional[Union[int, str, BaseCrossValidator]]
            The cross-validation strategy for computing conformity scores.
            It directly drives the distinction between jackknife and cv variants.
            Choose among:

            - ``None``, to use the default 5-fold cross-validation
            - integer, to specify the number of folds.
            If equal to ``-1``, equivalent to
            ``sklearn.model_selection.LeaveOneOut()``.
            - CV splitter: any ``sklearn.model_selection.BaseCrossValidator``
            Main variants are:
                - ``sklearn.model_selection.LeaveOneOut`` (jackknife),
                - ``sklearn.model_selection.KFold`` (cross-validation),
                - ``subsample.Subsample`` object (bootstrap).
            - ``"split"``, does not involve cross-validation but a division
            of the data into training and calibration subsets. The splitter
            used is the following: ``sklearn.model_selection.ShuffleSplit``.
            ``method`` parameter is set to ``"base"``.
            - ``"prefit"``, assumes that ``estimator`` has been fitted already,
            and the ``method`` parameter is set to ``"base"``.
            All data provided in the ``fit`` method is then used
            for computing conformity scores only.
            At prediction time, quantiles of these conformity scores are used
            to provide a prediction interval with fixed width.
            The user has to take care manually that data for model fitting and
            conformity scores estimate are disjoint.

            By default ``None``.

        
        Notes
        -----
        If more than one model is provided, predictions are made for both 
        invidiual models and an ensemble of the models. 

        For MAPIE: A GammaConformityScore assumes there is no zeros or negative values in Y.
        If this is not the case,  AbsoluteConformityScore() should be used.
        """

        number_of_models = len(self.ensemble_config) -2
        print("number of models in ensemble:" + str(number_of_models))

        if number_of_models==1:

            m, mae1 = def_prediction(self.path_out, self.ensemble_config, 0, self.target_no_space)

            if self.ensemble_config["regressor"] ==True:
                if prediction_inference==True:
                    mapie = MapieRegressor(m, conformity_score=conformity_score, cv=cv) #            
            else:
                if prediction_inference==True:
                    mapie = MapieClassifier(m, conformity_score=conformity_score, cv=cv) #


            model_name = self.ensemble_config["m" + str(1)]
            model_out = self.path_out + "predictions/" + model_name + "/" 
            export_prediction(m, self.target_no_space, self.X_predict, 
                              self.model_config, self.ensemble_config, 
                              model_out, n_threads=self.n_jobs)

        elif number_of_models >=2:
                    
            # iteratively make prediction for each model
            models = []
            mae_values = []
            w = []

            for i in range(number_of_models):
                m, mae = def_prediction(self.path_out, self.ensemble_config, i, self.target_no_space)
                model_name = self.ensemble_config["m" + str(i + 1)]
                model_out = self.path_out + "predictions/" + model_name + "/"  
                export_prediction(m, self.target_no_space, self.X_predict, 
                                  self.model_config, self.ensemble_config, 
                                  model_out, n_threads=self.n_jobs)

                print("exporting " + model_name + " prediction to: " + model_out)

                models.append((model_name, m))
                mae_values.append(mae)

            w = inverse_weighting(mae_values) 

            if self.ensemble_config["regressor"] ==True:
                m = VotingRegressor(estimators=models, weights=w).fit(self.X_train, self.y)
                if prediction_inference==True:
                    mapie = MapieRegressor(m, conformity_score=conformity_score, cv=cv) #            
            else:
                m= VotingClassifier(estimators=models, weights=w, voting='soft').fit(self.X_train, self.y)
                if prediction_inference==True:
                    mapie = MapieClassifier(m, cv=cv) #

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


        if prediction_inference==True:
            print()
            mapie.fit(self.X_train, self.y)

            #low_name = str(int(alpha[0]*100))
            pi_name = str(int(np.round((1-alpha[0])*100)))
            pi_LL = pi_name + "_LL"
            pi_UL = pi_name + "_UL"

            y_pred, y_pis = parallel_predict_mapie(self.X_predict, mapie, 
                                                    alpha, chunksize = 1000)

            low_model_out = self.path_out + "predictions/mapie/" + pi_LL +"/"
            pi50_model_out = self.path_out + "predictions/mapie/50/"
            up_model_out = self.path_out + "predictions/mapie/" + pi_UL +"/"
            
            try: #make new dir if needed
                os.makedirs(low_model_out)
            except:
                None
            try: #make new dir if needed
                os.makedirs(pi50_model_out)
            except:
                None
            try: #make new dir if needed
                os.makedirs(up_model_out)
            except:
                None

            d_pi50 = self.X_predict.copy()
            d_pi50[self.target] = y_pred
            d_pi50 = d_pi50.to_xarray()
            d_pi50[self.target].to_netcdf(pi50_model_out + self.target_no_space + ".nc", mode='w') 
            print("exported MAPIE PI50 prediction to: " + pi50_model_out + self.target_no_space + ".nc")
            d_pi50 = None
            y_pred = None

            d_low = self.X_predict.copy()

            d_low[self.target] = y_pis[:,0,:].flatten()
            d_low[self.target].to_xarray().to_netcdf(low_model_out + self.target_no_space + ".nc", mode='w') 
            print("exported MAPIE PI" + pi_LL + " prediction to: " + low_model_out + self.target_no_space + ".nc")
            d_low = None

            d_up = self.X_predict.copy()
            d_up[self.target] = y_pis[:,1,:].flatten()
            d_up[self.target].to_xarray().to_netcdf(up_model_out + self.target_no_space + ".nc", mode='w') 
            print("exported MAPIE PI" + pi_UL + " prediction to: " + up_model_out + self.target_no_space + ".nc")
            d_up = None

        et = time.time()
        elapsed_time = et-self.st
        print("finished")
        print("execution time:", elapsed_time, "seconds")