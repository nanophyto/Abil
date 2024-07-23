import unittest
unittest.TestLoader.sortTestMethodsUsing = None


import sys, os
from yaml import load
from yaml import CLoader as Loader

import pandas as pd
import pickle

from abil.tune import tune
from abil.functions import example_data, upsample
from abil.predict import predict


# class TestClassifiers(unittest.TestCase):

#     def test_tune_randomforest(self):
#         yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))

#         with open(yaml_path +'/configuration/classifier_test.yml', 'r') as f:
#             model_config = load(f, Loader=Loader)

#         X_train, y = example_data("Test")

#         m = tune(X_train, y, model_config)
    
#         m.train(model="rf", classifier=True)


#     def test_tune_xgb(self):
#         yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))

#         with open(yaml_path +'/configuration/classifier_test.yml', 'r') as f:
#             model_config = load(f, Loader=Loader)

#         X_train, y = example_data("Test")

#         m = tune(X_train, y, model_config)
    
#         m.train(model="xgb", classifier=True)


#     def test_tune_knn(self):
#         yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))

#         with open(yaml_path +'/configuration/classifier_test.yml', 'r') as f:
#             model_config = load(f, Loader=Loader)

#         X_train, y = example_data("Test")

#         m = tune(X_train, y, model_config)
    
#         m.train(model="knn", classifier=True)


#     def test_predict_ensemble(self):
#         yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))
        
#         with open(yaml_path +'/configuration/classifier_test.yml', 'r') as f:
#             model_config = load(f, Loader=Loader)

#         X_train, y = example_data("Test")
#         X_predict = X_train
        
#         m = predict(X_train, y, X_predict, model_config)
#         m.make_prediction()


#     # def clear_tmp(self):
#     #     yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))
        
#     #     with open(yaml_path +'/configuration/classifier_test.yml', 'r') as f:
#     #         model_config = load(f, Loader=Loader)

#     #     os.rmdir(model_config['local_root'])
#     #     print("deleted:" + model_config['local_root'])





class TestRegressors(unittest.TestCase):

    def test_tune_randomforest(self):
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
    
        m.train(model="rf", regressor=True)

        self.assertTrue(os.path.exists('/home/runner/work/Abil/Abil/tests/ModelOutput/rf/scoring/Emiliania_huxleyi_reg.sav'), 
                        "RF scoring file was not created.")

        # # Print the expected path
        # expected_model_path = "/home/runner/work/Abil/Abil/tests/ModelOutput/rf/scoring/Emiliania_huxleyi_reg.sav"
        # print(f"Expected model path: {expected_model_path}")

        # # Print the existence of the file
        # print(f"Does file exist: {os.path.exists(expected_model_path)}")




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

        self.assertTrue(os.path.exists('/home/runner/work/Abil/Abil/tests/ModelOutput/xgb/scoring/Emiliania_huxleyi_reg.sav'), 
                        "XGBoost scoring file was not created.")
        

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

        self.assertTrue(os.path.exists('/home/runner/work/Abil/Abil/tests/ModelOutput/knn/scoring/Emiliania_huxleyi_reg.sav'), 
                        "KNN scoring file was not created.")
        

    def test_predict_ensemble(self):
        yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))
        
        with open(yaml_path +'/tests/regressor.yml', 'r') as f:
            model_config = load(f, Loader=Loader)

        model_config['local_root'] = yaml_path
        predictors = model_config['predictors']
        d = pd.read_csv(model_config['local_root'] + model_config['training'])
        target =  "Emiliania huxleyi"

        expected_model_path = "/home/runner/work/Abil/Abil/tests/ModelOutput/rf/scoring/Emiliania_huxleyi_reg.sav"

        with open(expected_model_path, 'rb') as f:
            test = pickle.load(f)

        #test = pickle.load(open(expected_model_path, 'rb'))

        print(test)

        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)

        X_train = d[predictors]
        y = d[target]

        X_predict = X_train
        
        m = predict(X_train, y, X_predict, model_config)
        m.make_prediction()


    # def clear_tmp(self):
    #     yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))
        
    #     with open(yaml_path +'/configuration/classifier_test.yml', 'r') as f:
    #         model_config = load(f, Loader=Loader)

    #     os.rmdir(model_config['local_root'])
    #     print("deleted:" + model_config['local_root'])



#if __name__ == '__main__':
#    unittest.main()

if __name__ == '__main__':
    # Create a test suite combining all test cases in order
    suite = unittest.TestSuite()

    # Add tests to the suite in the desired order
    suite.addTest(TestRegressors('test_tune_randomforest'))
    suite.addTest(TestRegressors('test_tune_xgb'))
    suite.addTest(TestRegressors('test_tune_knn'))
    suite.addTest(TestRegressors('test_predict_ensemble'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)
