import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import sys, os
from yaml import load
from yaml import CLoader as Loader
import pandas as pd
import numpy as np
import xarray as xr 

from abil.tune import tune
from abil.utils import example_data # example_training_data, example_predict_data
from abil.predict import predict
from abil.post import post

class TestRegressors(unittest.TestCase):

    def setUp(self):
        self.workspace = os.getenv('GITHUB_WORKSPACE', '.')
        with open(os.path.join(self.workspace,'tests/regressor.yml'), 'r') as f:
            self.model_config = load(f, Loader=Loader)

        self.model_config['local_root'] = self.workspace # yaml_path


        self.target_name =  "Emiliania huxleyi"
        self.X_train, self.X_predict, self.y = example_data(self.target_name, n_samples=1000, n_features=3, noise=0.1, train_to_predict_ratio=0.7, random_state=59)
#        self.X_predict = X_predict[predictors]


    def test_post_ensemble(self):
        m = tune(self.X_train, self.y, self.model_config)
        m.train(model="rf", log="yes")
        m.train(model="xgb", log="yes")
        m.train(model="knn", log="yes")

        m = predict(self.X_train, self.y, self.X_predict, self.model_config, n_jobs=self.model_config['n_threads'])
        m.make_prediction()

        # Load datasets
        rf = xr.open_dataset("./tests/ModelOutput/regressor/predictions/rf/Emiliania_huxleyi.nc")
        xgb = xr.open_dataset("./tests/ModelOutput/regressor/predictions/xgb/Emiliania_huxleyi.nc")  # Note: same path as rf?

        # Calculate sums
        rf_mean_sum = np.sum(rf['mean'])
        rf_ci95_UL_sum = np.sum(rf['ci95_UL'])
        xgb_mean_sum = np.sum(xgb['mean'])
        xgb_ci95_UL_sum = np.sum(xgb['ci95_UL'])

        print("======================")
        print("DEBUG OF PREDICT SUMS")
        print("======================")

        print(f"RF mean sum: {rf_mean_sum}")
        print(f"XGB mean sum: {xgb_mean_sum}")
        # Check if values are within same order of magnitude
#        if not np.isclose(rf_mean_sum, xgb_mean_sum, rtol=9):
#            raise AssertionError("Mean sums are not within an order of magnitude")
        
        print(f"RF ci95_UL sum: {rf_ci95_UL_sum}")
        print(f"XGB ci95_UL sum: {xgb_ci95_UL_sum}")
#        if not np.isclose(rf_ci95_UL_sum, xgb_ci95_UL_sum, rtol=9):
#            raise AssertionError("CI95 UL sums are not within an order of magnitude")

        targets = np.array([self.target_name])
        def do_post(statistic):
            m = post(self.X_train, self.y, self.X_predict, self.model_config, statistic, datatype="poc")
            #estimate aoa for each target and export to aoa.nc:
            m.estimate_applicability(drop_zeros=True)

            m.estimate_carbon("pg poc")

            m.total()

            m.merge_env()
            m.merge_obs("test",targets)

            m.export_ds("test")
            m.export_csv("test")

            vol_conversion = 1e3 #L-1 to m-3
            integ = m.integration(m, vol_conversion=vol_conversion)
            print(targets)
            integ.integrated_totals(targets)
            integ.integrated_totals(targets, monthly=True)

        do_post(statistic="mean")
        do_post(statistic="ci95_UL")
        do_post(statistic="ci95_LL")




class Test2Phase(unittest.TestCase):

    def setUp(self):
        self.workspace = os.getenv('GITHUB_WORKSPACE', '.')
        with open(os.path.join(self.workspace, 'tests/2-phase.yml'), 'r') as f:
            self.model_config = load(f, Loader=Loader)

        self.model_config['local_root'] = self.workspace # yaml_path


        self.target_name =  "Emiliania huxleyi"

        self.X_train, self.X_predict, self.y = example_data(self.target_name, n_samples=1000, n_features=3, noise=0.1, train_to_predict_ratio=0.7, random_state=59)
        

    def test_post_ensemble(self):


        m = tune(self.X_train, self.y, self.model_config)

        m.train(model="rf", log="yes")
        m.train(model="xgb", log="yes")
        m.train(model="knn", log="yes")

        m = predict(self.X_train, self.y, self.X_predict, self.model_config, n_jobs=self.model_config['n_threads'])
        m.make_prediction()
        print("======================")
        print("DEBUG OF PREDICT SUMS")
        print("======================")
        # Load datasets
        rf = xr.open_dataset("./tests/ModelOutput/2-phase/predictions/rf/Emiliania_huxleyi.nc")
        xgb = xr.open_dataset("./tests/ModelOutput/2-phase/predictions/xgb/Emiliania_huxleyi.nc")  # Note: same path as rf?

        # Calculate sums
        rf_mean_sum = np.sum(rf['mean'])
        rf_ci95_UL_sum = np.sum(rf['ci95_UL'])
        xgb_mean_sum = np.sum(xgb['mean'])
        xgb_ci95_UL_sum = np.sum(xgb['ci95_UL'])

        print(f"RF mean sum: {rf_mean_sum}")
        print(f"XGB mean sum: {xgb_mean_sum}")
        
        # Check if values are within same order of magnitude
#        if not np.isclose(rf_mean_sum, xgb_mean_sum, rtol=9):
#            raise AssertionError("Mean sums are not within an order of magnitude")
        
        print(f"RF ci95_UL sum: {rf_ci95_UL_sum}")
        print(f"XGB ci95_UL sum: {xgb_ci95_UL_sum}")
#        if not np.isclose(rf_ci95_UL_sum, xgb_ci95_UL_sum, rtol=9):
#            raise AssertionError("CI95 UL sums are not within an order of magnitude")

        targets = np.array([self.target_name])

        def do_post(statistic):
            m = post(self.X_train, self.y, self.X_predict, self.model_config, statistic, datatype="poc")
            #estimate aoa for each target and export to aoa.nc:
            m.estimate_applicability(drop_zeros=True)
            m.estimate_carbon("pg poc")
            m.diversity()

            m.total()
            m.merge_env()
            m.merge_obs("test",targets)

            m.export_ds("test")

            vol_conversion = 1e3 #L-1 to m-3
            integ = m.integration(m, vol_conversion=vol_conversion)
            integ.integrated_totals(targets, monthly=True)
            integ.integrated_totals(targets)

        do_post(statistic="mean")
        do_post(statistic="ci95_UL")
        do_post(statistic="ci95_LL")


def _lat_band_area(lat_deg, dlat_deg):
    """R^2 * (sin(phi+Δφ/2) - sin(phi-Δφ/2))  -> shape (nlat,)"""
    phi = np.deg2rad(np.asarray(lat_deg))
    dphi = np.deg2rad(float(dlat_deg))
    return (6_371_000.0**2) * (np.sin(phi + dphi/2.0) - np.sin(phi - dphi/2.0))

def _horiz_area_lat_lon(lat_deg, lon_deg, dlat_deg, dlon_deg):
    """Per-cell horizontal area for (lat, lon): A_latband(lat) * Δλ  -> (nlat, nlon)"""
    lat_band = _lat_band_area(lat_deg, dlat_deg)     # (nlat,)
    dlam = np.deg2rad(float(dlon_deg))
    lon_width = np.full(len(lon_deg), dlam)          # (nlon,)
    return np.outer(lat_band, lon_width)             # (nlat, nlon)

class _IntegrationParent:
    """Minimal parent to satisfy the integration class."""
    def __init__(self, ds: xr.Dataset):
        self.d = ds.to_dataframe()
        self.root = "."
        self.model_config = {"path_out": ".", "run_name": "testrun"}
        self.statistic = "stat"
        self.datatype = ""

class TestIntegrationFlexibleDims(unittest.TestCase):
    """Checks known totals for multiple dimension combos."""

    def setUp(self):
        # tiny grid & constants
        self.lat = np.array([-0.5, 0.5])       # 1° tall bands around equator
        self.lon = np.array([0.0, 1.0, 2.0])   # 1° wide cells
        self.depth = np.array([5.0, 15.0])     # two layers (indices only)
        self.time = np.array([1, 2], dtype=int)  # integer months (Jan=1, Feb=2)
        self.dlat = 1.0
        self.dlon = 1.0
        self.depth_w = 10.0  # m per depth cell
        self.value = 2.0     # uniform field

        # expected helpers
        self.area2d = _horiz_area_lat_lon(self.lat, self.lon, self.dlat, self.dlon)
        self.horiz_total = self.area2d.sum()
        self.depth_total = self.depth_w * len(self.depth)
        self.zonal_area_by_lat = (2.0 * np.pi) * _lat_band_area(self.lat, self.dlat)
        self.zonal_total = self.zonal_area_by_lat.sum()

    def _make_integration(self, ds, rate=False):
        return post.integration(
            parent=_IntegrationParent(ds),
            resolution_lat=self.dlat,
            resolution_lon=self.dlon,
            depth_w=self.depth_w,
            rate=rate
        )

    def test_lat_lon_depth_time(self):
        coords = {"lat": self.lat, "lon": self.lon, "depth": self.depth, "time": self.time}
        data = np.full((len(self.lat), len(self.lon), len(self.depth), len(self.time)), self.value)
        ds = xr.Dataset({"foo": xr.DataArray(data, dims=("lat","lon","depth","time"), coords=coords)})

        integ = self._make_integration(ds, rate=False)
        total = integ.integrate_total(variable="foo", monthly=False)

        expected = self.value * self.horiz_total * self.depth_total * len(self.time)
        self.assertAlmostEqual(float(total.values), expected, places=10)

    def test_lat_lon_only(self):
        coords = {"lat": self.lat, "lon": self.lon}
        data = np.full((len(self.lat), len(self.lon)), self.value)
        ds = xr.Dataset({"foo": xr.DataArray(data, dims=("lat","lon"), coords=coords)})

        integ = self._make_integration(ds, rate=False)
        total = integ.integrate_total(variable="foo", monthly=False)

        expected = self.value * self.horiz_total
        self.assertAlmostEqual(float(total.values), expected, places=10)

    def test_lat_depth_only(self):
        coords = {"lat": self.lat, "depth": self.depth}
        data = np.full((len(self.lat), len(self.depth)), self.value)
        ds = xr.Dataset({"foo": xr.DataArray(data, dims=("lat","depth"), coords=coords)})

        integ = self._make_integration(ds, rate=False)
        total = integ.integrate_total(variable="foo", monthly=False)

        expected = self.value * self.zonal_total * self.depth_total
        self.assertAlmostEqual(float(total.values), expected, places=10)

    def test_time_only(self):
        coords = {"time": self.time}
        data = np.full((len(self.time),), self.value)
        ds = xr.Dataset({"foo": xr.DataArray(data, dims=("time",), coords=coords)})

        integ = self._make_integration(ds, rate=False)
        total = integ.integrate_total(variable="foo", monthly=False)

        expected = self.value * len(self.time)  # volume=1, just sum over time
        self.assertAlmostEqual(float(total.values), expected, places=10)


if __name__ == '__main__':
    # Create a test suite combining all test cases in order
    suite = unittest.TestSuite()
    suite.addTest(TestRegressors('test_post_ensemble'))
    suite.addTest(Test2Phase('test_post_ensemble'))

    # post area integration tests
    suite.addTest(TestIntegrationFlexibleDims('test_lat_lon_depth_time'))
    suite.addTest(TestIntegrationFlexibleDims('test_lat_lon_only'))
    suite.addTest(TestIntegrationFlexibleDims('test_lat_depth_only'))
    suite.addTest(TestIntegrationFlexibleDims('test_time_only'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
