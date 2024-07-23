import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import sys, os
from yaml import load
from yaml import CLoader as Loader
import pandas as pd
from abil.tune import tune
from abil.functions import example_data, upsample
from abil.predict import predict
from abil.post import post


class TestRegressors(unittest.TestCase):

    def setUp(self):
        self.workspace = os.getenv('GITHUB_WORKSPACE', '.')

    def test_tune_randomforest(self):

        with open(self.workspace +'/tests/regressor.yml', 'r') as f:
            model_config = load(f, Loader=Loader)

        model_config['local_root'] = self.workspace # yaml_path
        predictors = model_config['predictors']
        d = pd.read_csv(model_config['local_root'] + model_config['training'])
        target =  "Emiliania huxleyi"
        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)
        X_train = d[predictors]
        y = d[target]

        m = tune(X_train, y, model_config)
    
        m.train(model="rf", regressor=True)


    def test_tune_xgb(self):
        yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))

        with open(yaml_path +'/tests/regressor.yml', 'r') as f:
            model_config = load(f, Loader=Loader)

        model_config['local_root'] = yaml_path
        predictors = model_config['predictors']
        d = pd.read_csv(model_config['local_root'] + model_config['training'])
        target =  "Emiliania huxleyi"
        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)
        X_train = d[predictors]
        y = d[target]

        m = tune(X_train, y, model_config)
    
        m.train(model="xgb", regressor=True)

    def test_tune_knn(self):
        yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))

        with open(yaml_path +'/tests/regressor.yml', 'r') as f:
            model_config = load(f, Loader=Loader)

        model_config['local_root'] = yaml_path
        predictors = model_config['predictors']
        d = pd.read_csv(model_config['local_root'] + model_config['training'])
        target =  "Emiliania huxleyi"
        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)
        X_train = d[predictors]
        y = d[target]

        m = tune(X_train, y, model_config)
    
        m.train(model="knn", regressor=True)


    def test_predict_ensemble(self):
        self.test_tune_randomforest()
        self.test_tune_xgb()
        self.test_tune_knn()

        yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))
        
        with open(yaml_path +'/tests/regressor.yml', 'r') as f:
            model_config = load(f, Loader=Loader)

        model_config['local_root'] = yaml_path
        predictors = model_config['predictors']
        d = pd.read_csv(model_config['local_root'] + model_config['training'])
        target =  "Emiliania huxleyi"

        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)

        X_train = d[predictors]
        y = d[target]

        X_predict = X_train
        
        m = predict(X_train, y, X_predict, model_config)
        m.make_prediction()

    def test_post_ensemble(self):

        self.test_predict_ensemble()

        with open('./examples/configuration/regressor.yml', 'r') as f:
            model_config = load(f, Loader=Loader)

        model_config['hpc']==False

        predictors = model_config['predictors']
        d = pd.read_csv("./examples/data/training.csv")
        d.dropna(subset='FID', inplace=True)
        X_train = d[predictors]
        X_predict = pd.read_csv("./examples/data/prediction.csv")
        X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
        X_predict = X_predict[X_train.columns]

        m = post(model_config)
        m.merge_performance(model="ens")
        m.merge_performance(model="xgb", configuration= "reg")
        m.merge_performance(model="rf", configuration= "reg")
        m.merge_performance(model="knn", configuration= "reg")

        m.merge_parameters(model="rf")
        m.merge_parameters(model="xgb")
        m.merge_parameters(model="knn")

        m.total()

        m.merge_env(X_predict)

        m.export_ds("test")
        m.export_csv("test")


if __name__ == '__main__':
    # Create a test suite combining all test cases in order
    suite = unittest.TestSuite()

    # Add tests to the suite in the desired order
    suite.addTest(TestRegressors('test_tune_randomforest'))
    suite.addTest(TestRegressors('test_tune_xgb'))
    suite.addTest(TestRegressors('test_tune_knn'))
    suite.addTest(TestRegressors('test_predict_ensemble'))
    suite.addTest(TestRegressors('test_post_ensemble'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)
