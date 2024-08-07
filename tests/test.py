import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import sys, os
from yaml import load
from yaml import CLoader as Loader
import pandas as pd
from abil.tune import tune
from abil.functions import upsample, OffsetGammaConformityScore
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
        targets = pd.read_csv(self.model_config['local_root']+ self.model_config['targets'])
        n_spp = 0
        target =  targets['Target'][n_spp]
        d[target] = d[target].fillna(0)
        d = upsample(d, target, ratio=10)
        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)
        self.X_train = d[predictors]
        self.y = d[target]

        X_predict = pd.read_csv(self.model_config['local_root'] + self.model_config['prediction'])
        X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
        self.X_predict = X_predict[predictors]


    def test_post_ensemble(self):
        m = tune(self.X_train, self.y, self.model_config)
        m.train(model="rf", regressor=True)
        m.train(model="xgb", regressor=True)
        m.train(model="knn", regressor=True)

        m = predict(self.X_train, self.y, self.X_predict, self.model_config)
        m.make_prediction(prediction_inference=True)
        targets = pd.read_csv(self.model_config['local_root']+ self.model_config['targets'])
        targets = targets.iloc[:1]
        targets = targets['Target'].values

        def do_post(pi):
            m = post(self.model_config, pi=pi)
            m.merge_performance(model="ens") 
            m.merge_performance(model="xgb")
            m.merge_performance(model="rf")
            m.merge_performance(model="knn")

            m.merge_parameters(model="rf")
            m.merge_parameters(model="xgb")
            m.merge_parameters(model="knn")
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
            integ.integrated_totals(targets, subset_depth=100)

        do_post(pi="50")
        do_post(pi="95_UL")
        do_post(pi="95_LL")




class Test2Phase(unittest.TestCase):

    def setUp(self):
        self.workspace = os.getenv('GITHUB_WORKSPACE', '.')
        with open(self.workspace +'/tests/2-phase.yml', 'r') as f:
            self.model_config = load(f, Loader=Loader)

        self.model_config['local_root'] = self.workspace # yaml_path
        predictors = self.model_config['predictors']
        d = pd.read_csv(self.model_config['local_root'] + self.model_config['training'])
        targets = pd.read_csv(self.model_config['local_root']+ self.model_config['targets'])
        n_spp = 0
        target =  targets['Target'][n_spp]
        d[target] = d[target].fillna(0)
        d = upsample(d, target, ratio=10)
        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)
        self.X_train = d[predictors]
        self.y = d[target]

        X_predict = pd.read_csv(self.model_config['local_root'] + self.model_config['prediction'])
        X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
        self.X_predict = X_predict[predictors]

    def test_post_ensemble(self):


        m = tune(self.X_train, self.y, self.model_config)

        m.train(model="rf", classifier=True, regressor=True)
        m.train(model="xgb", classifier=True, regressor=True)
        m.train(model="knn", classifier=True, regressor=True)

        m = predict(self.X_train, self.y, self.X_predict, self.model_config)
        m.make_prediction(prediction_inference=True)
        targets = pd.read_csv(self.model_config['local_root']+ self.model_config['targets'])
        targets = targets.iloc[:1]
        targets = targets['Target'].values

        def do_post(pi):
            m = post(self.model_config, pi=pi)
            m.merge_performance(model="ens") 
            m.merge_performance(model="xgb")
            m.merge_performance(model="rf")
            m.merge_performance(model="knn")

            m.merge_parameters(model="rf")
            m.merge_parameters(model="xgb")
            m.merge_parameters(model="knn")
            m.estimate_carbon("pg poc")

            m.total()

            m.merge_env(self.X_predict)
            m.merge_obs("test",targets)


            m.export_ds("test")
            m.export_csv("test")

            vol_conversion = 1e3 #L-1 to m-3
            integ = m.integration(m, vol_conversion=vol_conversion)
            integ.integrated_totals(targets)
            integ.integrated_totals(targets, subset_depth=100)

        do_post(pi="50")
        do_post(pi="95_UL")
        do_post(pi="95_LL")



class TestClassifiers(unittest.TestCase):

    def setUp(self):
        self.workspace = os.getenv('GITHUB_WORKSPACE', '.')
        with open(self.workspace +'/tests/classifier.yml', 'r') as f:
            self.model_config = load(f, Loader=Loader)

        self.model_config['local_root'] = self.workspace # yaml_path
        predictors = self.model_config['predictors']
        d = pd.read_csv(self.model_config['local_root'] + self.model_config['training'])
        targets = pd.read_csv(self.model_config['local_root']+ self.model_config['targets'])
        n_spp = 0
        target =  targets['Target'][n_spp]
        d[target] = d[target].fillna(0)
        d = upsample(d, target, ratio=10)
        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)

        self.X_train = d[predictors]
        self.y = d[target]

        X_predict = pd.read_csv(self.model_config['local_root'] + self.model_config['prediction'])
        X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
        self.X_predict = X_predict[predictors]

    def test_post_ensemble(self):
   
        m = tune(self.X_train, self.y, self.model_config)
        m.train(model="rf", classifier=True)
        m.train(model="xgb", classifier=True)
        m.train(model="knn", classifier=True)

        m = predict(self.X_train, self.y, self.X_predict, self.model_config)
        m.make_prediction(prediction_inference=True)

        def do_post(pi):
            m = post(self.model_config, pi=pi)
            m.merge_performance(model="ens") 
            m.merge_performance(model="xgb")
            m.merge_performance(model="rf")
            m.merge_performance(model="knn")

            m.merge_parameters(model="rf")
            m.merge_parameters(model="xgb")
            m.merge_parameters(model="knn")
  
            m.estimate_carbon("pg poc")
            m.merge_env(self.X_predict)

            m.export_ds("test")
            m.export_csv("test")

        do_post(pi="50")
        do_post(pi="95_UL")
        do_post(pi="95_LL")



class TestGammaOffset(unittest.TestCase):

    def setUp(self):
        self.workspace = os.getenv('GITHUB_WORKSPACE', '.')
        with open(self.workspace +'/tests/regressor.yml', 'r') as f:
            self.model_config = load(f, Loader=Loader)

        self.model_config['local_root'] = self.workspace # yaml_path
        predictors = self.model_config['predictors']
        d = pd.read_csv(self.model_config['local_root'] + self.model_config['training'])
        target =  "Emiliania huxleyi"
        d[target] = d[target].fillna(0)
        d = upsample(d, target, ratio=10)
        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)
        self.X_train = d[predictors]
        self.y = d[target]

        X_predict = pd.read_csv(self.model_config['local_root'] + self.model_config['prediction'])
        X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
        self.X_predict = X_predict[predictors]


    def test_post_ensemble(self):
        m = tune(self.X_train, self.y, self.model_config)
        m.train(model="rf", regressor=True)
        m.train(model="xgb", regressor=True)
        m.train(model="knn", regressor=True)

        m = predict(self.X_train, self.y, self.X_predict, self.model_config)

        m.make_prediction(prediction_inference=True, 
                        conformity_score=OffsetGammaConformityScore(offset=1e-10))
        


if __name__ == '__main__':
    # Create a test suite combining all test cases in order
    suite = unittest.TestSuite()
    suite.addTest(TestClassifiers('test_post_ensemble'))
    suite.addTest(TestRegressors('test_post_ensemble'))
    suite.addTest(Test2Phase('test_post_ensemble'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
