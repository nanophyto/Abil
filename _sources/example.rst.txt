Examples
==========


1-phase random forest classifier
---------------------------------

setup
***********************

load packages:

>>> import pandas as pd
>>> import numpy as np
>>> from yaml import load
>>> from yaml import CLoader as Loader
>>> from planktonsdm.tune import tune
>>> from planktonsdm.predict import predict
>>> from planktonsdm.post import post
>>> from planktonsdm.functions import example_data

load configuration yaml:

>>> with open('/home/phyto/planktonSDM/configuration/example_model_config.yml', 'r') as f:
...	model_config = load(f, Loader=Loader)



>>> print(model_config)
  {'root': '/home/phyto/CoccoML/', 
  'path_out': 'ModelOutput/', 
  'traits': '/home/phyto/CoccoML/data/traits.csv', 
  'scale_X': True, 
  'verbose': 1, 
  'seed': 1, 
  'n_threads': 2, 
  'cv': 3, 
  'predict_probability': False, 
  'ensemble_config': {
  	'regressor': False, 
  	'classifier': True, 
  	'm1': 'rf'}, 
  'upsample': False, 
  'clf_scoring': {
  	'accuracy': 
  	'balanced_accuracy'}
  'param_grid': {
  	'rf_param_grid': {
  		'clf_param_grid': {
  			'n_estimators': [100], 
  			'max_features': [3, 4], 
  			'max_depth': [50, 100], 
  			'min_samples_leaf': [0.5, 1], 
  			'max_samples': [0.5, 1]}}}
  }


  
create example count data:

>>> X, y = example_data(y_name =  "Coccolithus pelagicus", n_samples=500, n_features=5, 
...	noise=20, random_state=model_config['seed'])

create example envdata

>>> envdata = pd.DataFrame({"no3": rand(50), "mld": rand(50), "par": rand(50), "o2": rand(50), 
...	"temp": rand(50), "lat": range(0,50, 1), "lon": 50*[1]})
>>> envdata.set_index(['lat', 'lon'], inplace=True)


train
***********************
>>> m = tune(X, y, model_config)
>>> m.train(model="rf", classifier=True)

predict
***********************
>>> envdata = pd.DataFrame(X)
>>> m = predict(X, y, envdata, model_config)
>>> m.make_prediction()


post
***********************
>>> m = post(model_config)
>>> m.merge_env()
>>> m.export_ds(file_name = "1-phase_rf")


2-phase XGBoost regressor
---------------------------

setup
***********************

load configuration yaml:

>>> with open('/home/phyto/planktonSDM/configuration/example_model_config.yml', 'r') as f:
...	model_config = load(f, Loader=Loader)

>>> print(model_config)
  {'root': '/home/phyto/CoccoML/', 
  'path_out': 'ModelOutput/', 
  'traits': '/home/phyto/CoccoML/data/traits.csv', 
  'scale_X': True, 
  'verbose': 1, 
  'seed': 1, 
  'n_threads': 2, 
  'cv': 3, 
  'predict_probability': False, 
  'ensemble_config': {
  	'regressor': True, 
  	'classifier': True, 
  	'm2': 'xgb'}, 
  'upsample': False, 
  'clf_scoring': {
  	'accuracy': 
  	'balanced_accuracy'}, 
  'reg_scoring': {
  	'R2': 'r2', 
  	'MAE': 'neg_mean_absolute_error', 
  	'RMSE': 'neg_root_mean_squared_error'}, 
  'param_grid': {
  	'xgb_param_grid': {
  		'clf_param_grid': {
  			'eta': [0.01], 
  			'n_estimators': [100], 
  			'max_depth': [4], 
  			'subsample': [0.6], 
  			'colsample_bytree': [0.6], 
  			'gamma': [1], 
  			'alpha': [1]}, 
  		'reg_param_grid': {
  			'regressor__eta': [0.01], 
  			'regressor__n_estimators': [100], 
  			'regressor__max_depth': [4], 
  			'regressor__subsample': [0.6], 
  			'regressor__colsample_bytree': [0.6], 
  			'regressor__gamma': [1], 
  			'regressor__alpha': [1]}}}, 


train
***********************
>>> m = tune(X, y, model_config)
>>> m.train(model="xgb", classifier=True, regressor=True)


predict
***********************
>>> envdata = pd.DataFrame(X)
>>> m = predict(X, y, envdata, model_config)
>>> m.make_prediction()


post
***********************
>>> m = post(model_config)
>>> m.merge_env()
>>> m.export_ds(file_name = "2-phase_xgboost")





1-phase ensemble regression
------------------------------

setup
***********************

load configuration yaml:

>>> with open('/home/phyto/planktonSDM/configuration/example_model_config.yml', 'r') as f:
...	model_config = load(f, Loader=Loader)

>>> print(model_config)
  {'root': '/home/phyto/CoccoML/', 
  'path_out': 'ModelOutput/', 
  'traits': '/home/phyto/CoccoML/data/traits.csv', 
  'scale_X': True, 
  'verbose': 1, 
  'seed': 1, 
  'n_threads': 2, 
  'cv': 3, 
  'predict_probability': False, 
  'ensemble_config': {
  	'regressor': False, 
  	'classifier': True, 
  	'm1': 'rf', 
  	'm2': 'xgb', 
  	'm3': 'knn'}, 
  'upsample': False, 
  'reg_scoring': {
  	'R2': 'r2', 
  	'MAE': 'neg_mean_absolute_error', 
  	'RMSE': 'neg_root_mean_squared_error'}, 
  'param_grid': {
  	'rf_param_grid': {
  		'reg_param_grid': {
  			'regressor__n_estimators': [100], 
  			'regressor__max_features': [3, 4], 
  			'regressor__max_depth': [50, 100], 
  			'regressor__min_samples_leaf': [0.5, 1], 
  			'regressor__max_samples': [0.5, 1]}, 
  	'xgb_param_grid': {
  		'reg_param_grid': {
  			'regressor__eta': [0.01], 
  			'regressor__n_estimators': [100], 
  			'regressor__max_depth': [4], 
  			'regressor__subsample': [0.6], 
  			'regressor__colsample_bytree': [0.6], 
  			'regressor__gamma': [1], 
  			'regressor__alpha': [1]}}, 
  	'knn_param_grid': {
  		'reg_param_grid': {
  			'regressor__max_samples': [0.5], 
  			'regressor__max_features': [0.5], 
  			'regressor__estimator__leaf_size': [30], 
  			'regressor__estimator__n_neighbors': [3], 
  			'regressor__estimator__p': [1], 
  			'regressor__estimator__weights': ['uniform']}}}, 	
  'knn_bagging_estimators': 30}


train
***********************
>>> m = tune(X, y, model_config)
>>> m.train(model="rf", regressor=True)
>>> m.train(model="knn", regressor=True)
>>> m.train(model="xgb", regressor=True)


predict
***********************
>>> envdata = pd.DataFrame(X)
>>> m = predict(X, y, envdata, model_config)
>>> m.make_prediction()


post
***********************
>>> m = post(model_config)
>>> m.merge_env()
>>> m.export_ds(file_name = "1-phase_ensemble_regression")





