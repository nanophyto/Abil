import unittest
import os
import pandas as pd
from yaml import load
from yaml import CLoader as Loader
from abil.tune import tune
from abil.predict import predict
from abil.post import post
from abil.functions import upsample 

unittest.TestLoader.sortTestMethodsUsing = None

class BaseTestModel(unittest.TestCase):

    def setUp(self, config_file):
        self.workspace = os.getenv('GITHUB_WORKSPACE', '.')
        with open(f'{self.workspace}/tests/{config_file}', 'r') as f:
            self.model_config = load(f, Loader=Loader)
        self.model_config['local_root'] = self.workspace

        predictors = self.model_config['predictors']
        data = pd.read_csv(f"{self.model_config['local_root']}{self.model_config['training']}")
        target = "Emiliania huxleyi"
        data[target] = data[target].fillna(0)
        data = upsample(data, target, ratio=10)
        data = data.dropna(subset=[target] + predictors)
        self.X_train = data[predictors]
        self.y = data[target]

        X_predict = pd.read_csv(f"{self.model_config['local_root']}{self.model_config['prediction']}")
        X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
        self.X_predict = X_predict[predictors]

    def tune_and_train(self, model, **kwargs):
        m = tune(self.X_train, self.y, model_config = self.model_config)
        m.train(model=model, **kwargs)

    def test_tune_rf(self):
        self.tune_and_train("rf", **self.model_params)

    def test_tune_xgb(self):
        self.tune_and_train("xgb", **self.model_params)

    def test_tune_knn(self):
        self.tune_and_train("knn", **self.model_params)

    def test_predict_ensemble(self):
        self.test_tune_rf()
        self.test_tune_xgb()
        self.test_tune_knn()
        m = predict(self.X_train, self.y, self.X_predict, self.model_config)
        m.make_prediction(prediction_inference=True)

    def test_post_ensemble(self):
        self.test_predict_ensemble()

        print(f"{self.model_config['local_root']}{self.model_config['path_out']}")
        print("checking post predictions:")
        print(os.listdir(f"{self.model_config['local_root']}{self.model_config['path_out']}"))

        m = post(self.model_config)
        models = ["ens", "xgb", "rf", "knn"]
        for model in models:
            m.merge_performance(model=model, configuration="reg" if model != "ens" else None)
            if model != "ens":
                m.merge_parameters(model=model)

        try:
            if self.model_params['regressor']:
                m.total()
        except:
            pass

        m.merge_env(self.X_predict)
        m.export_ds("test")
        m.export_csv("test")

        try:
            if self.model_params['regressor']:
                targets = ['Emiliania huxleyi', 'total']
                vol_conversion = 1e3 #L-1 to m-3
                integ = m.integration(m, vol_conversion=vol_conversion)
                integ.integrated_totals(targets)
                integ.integrated_totals(targets, subset_depth=100)
        except:
            pass  # or handle the exception appropriately




class TestRegressors(BaseTestModel):

    @classmethod
    def setUpClass(cls):
        cls.model_params = {'regressor': True}

    def setUp(self):
        super().setUp(config_file='regressor.yml')


class TestClassifiers(BaseTestModel):

    @classmethod
    def setUpClass(cls):
        cls.model_params = {'classifier': True}

    def setUp(self):
        super().setUp(config_file='classifier.yml')


class Test2Phase(BaseTestModel):

    @classmethod
    def setUpClass(cls):
        cls.model_params = {'classifier': True, 'regressor': True}

    def setUp(self):
        super().setUp(config_file='2-phase.yml')


if __name__ == '__main__':
    # Create a test suite combining all test cases in order
    suite = unittest.TestSuite()
    suite.addTest(TestClassifiers('test_post_ensemble'))
    suite.addTest(TestRegressors('test_post_ensemble'))
#    suite.addTest(Test2Phase('test_post_ensemble'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
