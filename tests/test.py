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
        with open(self.workspace +'/tests/regressor.yml', 'r') as f:
            self.model_config = load(f, Loader=Loader)

        self.model_config['local_root'] = self.workspace # yaml_path
        predictors = self.model_config['predictors']
        d = pd.read_csv(self.model_config['local_root'] + self.model_config['training'])
        target =  "Emiliania huxleyi"
        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)
        self.X_train = d[predictors]
        self.y = d[target]

        X_predict = pd.read_csv(self.model_config['local_root'] + self.model_config['prediction'])
        X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
        self.X_predict = X_predict[predictors]


    def test_tune_randomforest(self):
        m = tune(self.X_train, self.y, self.model_config)
        m.train(model="rf", regressor=True)

    def test_tune_xgb(self):
        m = tune(self.X_train, self.y, self.model_config)
        m.train(model="xgb", regressor=True)

    def test_tune_knn(self):
        m = tune(self.X_train, self.y, self.model_config)
        m.train(model="knn", regressor=True)

    def test_predict_ensemble(self):
        self.test_tune_randomforest()
        self.test_tune_xgb()
        self.test_tune_knn()

        m = predict(self.X_train, self.y, self.X_predict, self.model_config)
        m.make_prediction(prediction_inference=True)


    def test_post_ensemble(self):

        self.test_predict_ensemble()

        print(self.model_config['local_root'] + self.model_config['path_out'])


        print("checking post predictions:")

        print(os.listdir(self.model_config['local_root'] + self.model_config['path_out']))

        m = post(self.model_config)
        m.merge_performance(model="ens") 
        m.merge_performance(model="xgb", configuration= "reg")
        m.merge_performance(model="rf", configuration= "reg")
        m.merge_performance(model="knn", configuration= "reg")

        m.merge_parameters(model="rf")
        m.merge_parameters(model="xgb")
        m.merge_parameters(model="knn")

        m.total()

        m.merge_env(self.X_predict)

        m.export_ds("test")
        m.export_csv("test")


if __name__ == '__main__':
    # Create a test suite combining all test cases in order
    suite = unittest.TestSuite()

    # Add tests to the suite in the desired order
#    suite.addTest(TestRegressors('test_tune_randomforest'))
#    suite.addTest(TestRegressors('test_tune_xgb'))
#    suite.addTest(TestRegressors('test_tune_knn'))
#    suite.addTest(TestRegressors('test_predict_ensemble'))
    suite.addTest(TestRegressors('test_post_ensemble'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)
