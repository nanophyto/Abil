import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import sys, os
from yaml import load
from yaml import CLoader as Loader
import pandas as pd

import numpy as np


if os.path.exists(os.path.join(os.path.dirname(__file__), '../.git')): #assumes this local
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../abil/')))
    from tune import tune
    from functions import upsample, example_data# example_training_data, example_predict_data
    from predict import predict
    from post import post
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
else: #if on github CI 
    from abil.tune import tune

    from abil.functions import upsample, example_data# example_training_data, example_predict_data
    from abil.predict import predict
    from abil.post import post

class TestRegressors(unittest.TestCase):

    def setUp(self):
        self.workspace = os.getenv('GITHUB_WORKSPACE', '.')
        with open(os.path.join(self.workspace,'tests/regressor.yml'), 'r') as f:
            self.model_config = load(f, Loader=Loader)
            
        self.model_config['local_root'] = self.workspace # yaml_path


        self.target_name =  "Emiliania huxleyi"
        self.X_train, self.X_predict, self.y = example_data(self.target_name, n_samples=200, n_features=3, noise=0.1, train_to_predict_ratio=0.7, random_state=59)
#        self.X_predict = X_predict[predictors]


    def test_post_ensemble(self):
        m = tune(self.X_train, self.y, self.model_config)
        m.train(model="rf")
        m.train(model="xgb")
        m.train(model="knn")

        m = predict(self.X_train, self.y, self.X_predict, self.model_config)
        m.make_prediction()

        targets = np.array([self.target_name])
        def do_post(pi):
            m = post(self.model_config, pi=pi)
            m.estimate_carbon("pg poc")

            m.total()

            m.merge_env(self.X_predict)
            m.merge_obs("test",targets)

            m.export_ds("test")
            m.export_csv("test")

            vol_conversion = 1e3 #L-1 to m-3
            integ = m.integration(m, vol_conversion=vol_conversion)
            print(targets)
            integ.integrated_totals(targets)
            integ.integrated_totals(targets, monthly=True)

        do_post(pi="50")




class Test2Phase(unittest.TestCase):

    def setUp(self):
        self.workspace = os.getenv('GITHUB_WORKSPACE', '.')
        with open(os.path.join(self.workspace, 'tests/2-phase.yml'), 'r') as f:
            self.model_config = load(f, Loader=Loader)

        self.model_config['local_root'] = self.workspace # yaml_path


        self.target_name =  "Emiliania huxleyi"

        self.X_train, self.X_predict, self.y = example_data(self.target_name, n_samples=200, n_features=3, noise=0.1, train_to_predict_ratio=0.7, random_state=59)


    def test_post_ensemble(self):


        m = tune(self.X_train, self.y, self.model_config)

        m.train(model="rf")
        m.train(model="xgb")
        m.train(model="knn")

        m = predict(self.X_train, self.y, self.X_predict, self.model_config)
        m.make_prediction()

        targets = np.array([self.target_name])

        def do_post(pi):
            m = post(self.model_config, pi=pi, datatype="poc")
            m.estimate_carbon("pg poc")
            m.diversity()

            m.total()
            m.merge_env(self.X_predict)
            m.merge_obs("test",targets)

            m.export_ds("test")
            m.export_csv("test")

            vol_conversion = 1e3 #L-1 to m-3
            integ = m.integration(m, vol_conversion=vol_conversion)
            integ.integrated_totals(targets, monthly=True)
            integ.integrated_totals(targets)


        do_post(pi="50")

if __name__ == '__main__':
    # Create a test suite combining all test cases in order
    suite = unittest.TestSuite()
    suite.addTest(TestRegressors('test_post_ensemble'))
    suite.addTest(Test2Phase('test_post_ensemble'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
