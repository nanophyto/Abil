"""Focused tests for Abil's permutation-importance analysis."""
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from abil.analyze import area_of_applicability, permutation_feature_importance
from abil.post import AbilPostProcessor
from abil.zir import ZeroInflatedRegressor


class TestFeatureImportance(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(10)
        self.X = pd.DataFrame(rng.normal(size=(60, 3)), columns=['temperature', 'nitrate', 'light'])
        self.y = pd.Series(4 * self.X['temperature'] + rng.normal(scale=.2, size=60), name='Target A')
        self.model = make_pipeline(StandardScaler(), Ridge()).fit(self.X, self.y)

    def test_importance_preserves_predictors_and_repeats(self):
        summary, repeats = permutation_feature_importance(
            self.model, self.X, self.y, cv=KFold(3, shuffle=True, random_state=1),
            n_repeats=4, random_state=2, return_repeats=True,
        )
        self.assertEqual(summary['feature'].tolist(), list(self.X.columns))
        self.assertEqual(len(repeats), 3 * 3 * 4)
        self.assertTrue((summary['n_folds'] == 3).all())
        self.assertGreater(summary['importance_mean'].iloc[0], 1)
        again = permutation_feature_importance(
            self.model, self.X, self.y, cv=KFold(3, shuffle=True, random_state=1),
            n_repeats=4, random_state=2,
        )
        assert_frame_equal(summary, again)

    def test_grouped_cv_and_prefit_estimator(self):
        groups = np.repeat(np.arange(3), 20)
        out = permutation_feature_importance(
            self.model, self.X, self.y, cv=GroupKFold(3), groups=groups,
            n_repeats=2, random_state=1,
        )
        self.assertTrue((out['n_folds'] == 3).all())
        prefit = permutation_feature_importance(
            self.model, self.X, self.y, n_repeats=2, random_state=1,
        )
        self.assertTrue((prefit['n_folds'] == 1).all())

    def test_area_of_applicability_handles_nonpositive_importances_and_weights(self):
        with patch('abil.analyze.permutation_feature_importance', return_value=pd.DataFrame(
            {'importance_mean': [-0.1, 0, -0.3]}
        )):
            aoa, di, cutoff = area_of_applicability(
                self.X.iloc[:8], self.X.iloc[8:], y_train=self.y.iloc[8:],
                model=self.model, feature_weight_kwargs={'n_jobs': 1},
            )
        self.assertEqual(aoa.shape, (8,))
        self.assertTrue(np.isfinite(di).all())
        self.assertTrue(np.isfinite(cutoff))
        area_of_applicability(self.X.iloc[:8], self.X.iloc[8:], feature_weights=np.ones(3))
        with self.assertRaises(ValueError):
            area_of_applicability(self.X.iloc[:8], self.X.iloc[8:], feature_weights=np.array([1, -1, 1]))

    def test_importance_uses_original_categorical_predictor(self):
        X = self.X.copy()
        X['region'] = np.where(X['temperature'] > 0, 'north', 'south')
        preprocessor = ColumnTransformer([
            ('number', StandardScaler(), ['temperature', 'nitrate', 'light']),
            ('region', OneHotEncoder(handle_unknown='ignore'), ['region']),
        ])
        model = make_pipeline(preprocessor, Ridge()).fit(X, self.y)
        result = permutation_feature_importance(model, X, self.y, n_repeats=2)
        self.assertEqual(result.feature.tolist(), list(X.columns))

    def test_postprocessor_exports_models_and_ensemble(self):
        with tempfile.TemporaryDirectory() as root:
            config = {
                'root': root, 'path_out': 'runs', 'run_name': 'trial',
                'seed': 2, 'cv': 3, 'stratify': False,
                'ensemble_config': {'classifier': False, 'regressor': True, 'm1': 'rf'},
            }
            model_dir = os.path.join(root, 'runs', 'trial', 'model')
            for model_name, model in [('rf', self.model),
                                      ('ens', VotingRegressor([('ridge', Ridge())]).fit(self.X, self.y))]:
                path = os.path.join(model_dir, model_name)
                os.makedirs(path)
                with open(os.path.join(path, 'Target_A_reg.sav'), 'wb') as file:
                    pickle.dump(model, file)

            post = AbilPostProcessor.__new__(AbilPostProcessor)
            post.X_train, post.y_train = self.X, self.y.to_frame()
            post.targets, post.model_config = ['Target A'], config
            post.root, post.extension = root, '_reg.sav'
            post.path_out = os.path.join(root, 'runs', 'trial', 'posts')
            summary, repeats = post.estimate_feature_importance(n_repeats=2)
            self.assertEqual(set(summary.model), {'rf', 'ens'})
            self.assertEqual(len(repeats), 2 * 3 * 3 * 2)
            self.assertTrue((summary['evaluation'] == 'cross_validation').all())
            self.assertTrue(os.path.isfile(os.path.join(post.path_out, 'feature_importance', 'summary.csv')))
            self.assertTrue(os.path.isfile(os.path.join(post.path_out, 'feature_importance', 'repeats.csv')))

            # Inspect only the final saved ensemble, without fitting another model.
            with patch('abil.analyze.clone', side_effect=AssertionError('unexpected refit')):
                fitted_summary, fitted_repeats = post.estimate_feature_importance(
                    models=['ens'], cv=False, n_repeats=2, export=False,
                )
            self.assertEqual(fitted_summary.model.tolist(), ['ens'] * 3)
            self.assertTrue((fitted_summary.evaluation == 'fitted').all())
            self.assertTrue((fitted_summary.n_folds == 1).all())
            self.assertEqual(len(fitted_repeats), 3 * 2)

    def test_two_phase_uses_final_concentration(self):
        rng = np.random.RandomState(4)
        X = pd.DataFrame(rng.normal(size=(48, 2)), columns=['x', 'z'])
        y = pd.Series(np.where(X['x'] > 0, 2 + X['x'], 0))
        model = ZeroInflatedRegressor(
            RandomForestClassifier(n_estimators=8, random_state=1),
            RandomForestRegressor(n_estimators=8, random_state=1),
        ).fit(X, y)
        out = permutation_feature_importance(
            model, X, y, cv=KFold(3, shuffle=True, random_state=1),
            n_repeats=2, random_state=1,
        )
        self.assertEqual(out.shape[0], 2)
        self.assertGreater(out.importance_mean.iloc[0], out.importance_mean.iloc[1])


if __name__ == '__main__':
    unittest.main()
