import pandas as pd
import numpy as np
import glob, os
import xarray as xr
import pickle
import gc
from yaml import dump, Dumper
from skbio.diversity.alpha import shannon


from .analyze import area_of_applicability

class post:
    """
    Post processing of SDM
    """
    def __init__(self, X_train, y_train, X_predict, model_config, statistic="mean", datatype=None):
        """
        A class for initializing and setting up a model with configuration, input data, and parameters.

        Parameters
        ----------
        X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training features used for model fitting.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values used for model fitting.
        X_predict : {array-like, sparse matrix} of shape (n_samples, n_features)
            Features to predict on (e.g., environmental data).
        model_config : dict
            Dictionary containing model configuration parameters such as:
            - seed: int, random seed for reproducibility
            - path_out: str, output path for saving results
            - path_in: str, input path to models
            - verbose: int, verbosity level (0-3)
            - cv: int, number of cross-validation folds
            - ensemble_config: dict, configuration for ensemble models
        pi : str
            The prediction interval identifier, defaulting to "50".
        datatype : str, optional
            The datatype of the predictions. This is used to access conversion factors (e.g. pic or poc) 
            and is appended to the file names during export            

        Attributes
        ----------
        path_out : str
            The output directory path where the model results will be saved.
        ds : xarray.Dataset
            The dataset containing the merged data from NetCDF files.
        traits : pd.DataFrame
            DataFrame containing the target trait information loaded from a CSV file.
        root : str
            Root directory as specified in the model configuration.
        d : pd.DataFrame
            DataFrame representation of the dataset after conversion and cleaning.
        targets : pd.Series
            The target values from the trait data that are present in the dataset columns.
        model_config : dict
            The model configuration dictionary containing paths, parameters, and other settings.
        pi : str
            The input parameter identifier, defaulting to "50".
        model_type : str
            The type of model being used, determined from the ensemble configuration (either "zir" or "reg").
        extension : str
            The file extension used for saving the model, based on the model type (e.g., "_zir.sav").
        datatype: str
            The datatype of the data being processed (e.g. "pg poc") which is appended to the data exports (optional)

        Methods
        -------
        merge_netcdf(path_in):
            Merges multiple NetCDF files from the specified directory into a single dataset.
        """

        def merge_netcdf(path_in, statistic):
            """
            Merges multiple NetCDF files from the specified directory into a single dataset.

            This function uses `xarray.open_mfdataset` to load all NetCDF files in the given directory 
            (matching the pattern "*.nc") and combines them into one xarray.Dataset. The function 
            prints status messages indicating the start and completion of the merging process.

            Parameters
            ----------
            path_in : str
                The path to the directory containing the NetCDF files to be merged.
            statistic : str
                The name of the statistic variable to extract from each dataset.

            Returns
            -------
            xarray.Dataset
                The merged dataset containing the combined data from all the NetCDF files in the directory.
                The variable names in the merged dataset are derived from the 'target' values in each file.
            """
            print("merging...")
            print(path_in)

            datasets = []
            
            for file in os.listdir(path_in):
                if file.endswith(".nc"):
                    ds = xr.open_dataset(os.path.join(path_in, file))
                    if statistic in ds:
                        # Extract the target name
                        target_name = ds['target'].values.item()  # Assuming target is a single value
                        
                        # Select the statistic and rename it to the target name
                        ds_subset = ds[[statistic]].rename({statistic: target_name})
                        datasets.append(ds_subset)
                    else:
                        print(f"Statistic '{statistic}' not found in {file}")

            # Merge datasets by variables, keeping same coordinates
            merged_ds = xr.merge(datasets, compat='override')  # 'override' skips conflicts

            print("finished merging NetCDF files")
            return merged_ds

        self.path_out = os.path.join(model_config['root'], model_config['path_out'], model_config['run_name'], "posts/")
        self.ds = merge_netcdf(os.path.join(model_config['root'], model_config['path_out'], model_config['run_name'], model_config['path_in']), statistic)
        self.traits = pd.read_csv(os.path.join(model_config['root'], model_config['targets']))

        self.root  =  model_config['root'] 
        self.statistic = statistic

        self.d = self.ds.to_dataframe()
        self.unique_targets = np.unique(self.d.columns.values).tolist()

        self.d = self.d.dropna()
        self.targets = self.unique_targets

        self.model_config = model_config

        self.y_train = y_train
        self.X_train = X_train
        self.X_predict = X_predict
   
        # Export model_config to a YAML file
        self.export_model_config()
        if self.model_config['ensemble_config']['classifier'] and not self.model_config['ensemble_config']['regressor']:
            raise ValueError("classifiers are not supported")
        elif self.model_config['ensemble_config']['classifier'] and self.model_config['ensemble_config']['regressor']:
            self.model_type = "zir"
        if self.model_config['ensemble_config']['regressor'] and not self.model_config['ensemble_config']['classifier']:
            self.model_type = "reg"

        self.extension = "_" + self.model_type + ".sav"

        self.merge_parameters()
        self.merge_performance()

        if datatype:
            self.datatype = "_" + datatype
        else:
            self.datatype = ""
        
        
    def export_model_config(self):
        """
        Export the model_config dictionary to a YAML file in self.path_out.
        
        Raises
        ------
        Exception
            If an error occurs during the directory creation or file writing process, an exception
            is caught and an error message is printed.

        Notes
        -----
        The YAML file is saved as "model_config.yml" in the `self.path_out` directory.
        """
        try:
            os.makedirs(self.path_out, exist_ok=True)  # Ensure the output directory exists
            yml_file_path = os.path.join(self.path_out, "model_config.yml")
            
            # Write the model_config dictionary to a YAML file
            with open(yml_file_path, 'w') as yml_file:
                dump(self.model_config, yml_file, Dumper=Dumper, default_flow_style=False)
            
            print(f"Model configuration exported to: {yml_file_path}")
        except Exception as e:
            print(f"Error exporting model_config to YAML: {e}")   

    def merge_performance(self):
        """
        Merges the performance data of multiple models as specified in the model configuration.

        Notes
        -----
        The function relies on the `merge_performance_single_model` method to merge individual model 
        performance data, and this is done for each model in the list, including the ensemble model.
        """    

        models = [value for key, value in self.model_config['ensemble_config'].items() if key.startswith("m")]
        print("models included in merge performance!")
        print(models)
        models.append("ens")
        for model in models:
            self.merge_performance_single_model(model)

       
    def merge_performance_single_model(self, model):
        """
        Merges performance metrics for a single model and saves the results to a CSV file.

        Parameters
        ----------
        model : str
            The name of the model for which performance metrics are being calculated and merged. 
            The model's performance data is expected to be stored in a `pickle` file in the "scoring" 
            directory under the model name and target name.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the model configuration includes a classifier but not a regressor, an error is raised 
            since classifiers are not supported for performance merging.

        Notes
        -----
        The method calculates several performance metrics for each target column in the dataset:
            - R2: Coefficient of determination.
            - RMSE: Root Mean Squared Error.
            - MAE: Mean Absolute Error.
            - rRMSE: Relative Root Mean Squared Error.
            - rMAE: Relative Mean Absolute Error.

        The performance metrics for each target are aggregated into a DataFrame, which is then saved 
        as a CSV file in the "posts/performance" directory for the specified model.
        """
        
        all_performance = []

        for i in range(len(self.unique_targets)):
            
            target = self.unique_targets[i]
            target_no_space = target.replace(' ', '_')
            with open(os.path.join(self.root, self.model_config['path_out'], self.model_config['run_name'], "scoring", model, target_no_space) + self.extension, 'rb') as file:

                m = pickle.load(file)
            
            if self.model_config['ensemble_config']['classifier'] and not self.model_config['ensemble_config']['regressor']:
                raise ValueError("classifiers are not supported")
            else:
                mean = np.mean(self.d[self.d.columns[i]])
                R2 = np.mean(m['test_R2'])
                RMSE = -1*np.mean(m['test_RMSE'])
                MAE = -1*np.mean(m['test_MAE'])
                rRMSE = -1*np.mean(m['test_RMSE'])/mean
                rMAE = -1*np.mean(m['test_MAE'])/mean            
                target = self.d.columns[i]
                performance = pd.DataFrame({'target':[target], 'R2':[R2], 'RMSE':[RMSE], 'MAE':[MAE],
                                            'rRMSE':[rRMSE], 'rMAE':[rMAE]})
                all_performance.append(performance)

        all_performance = pd.concat(all_performance)
        try: #make new dir if needed
            os.makedirs(os.path.join(self.root, self.model_config['path_out'], self.model_config['run_name'], "posts/performance"))
        except:
            None
        all_performance.to_csv(os.path.join(self.root, self.model_config['path_out'], self.model_config['run_name'], "posts/performance", model) + "_performance.csv", index=False)

        print("finished merging performance")

    def merge_parameters(self):
        """
        Merges model parameters for multiple models as specified in the model configuration.

        Notes
        -----
        The method operates by iterating over each model in the ensemble configuration, collecting 
        model parameters using `merge_parameters_single_model`, and saving the results to a CSV file 
        in the "posts/parameters" directory.
        """

        models = [value for key, value in self.model_config['ensemble_config'].items() if key.startswith("m")]
        for model in models:
            self.merge_parameters_single_model(model)

    def merge_parameters_single_model(self, model):
        """
        Merges and saves model parameters for a single model.

        This method extracts the hyperparameters of a specified model (e.g., "rf", "xgb", "knn") from 
        serialized files stored as pickle objects. The method supports different model types, including 
        regression ("reg"), classification ("clf"), and ensemble ("zir") models. The extracted parameters 
        are stored in a DataFrame and then saved to a CSV file.

        The function also handles the creation of the necessary directories to save the resulting CSV file 
        if they do not already exist.

        Parameters
        ----------
        model : str
            The name of the model for which parameters are being merged. Expected models include 
            "rf" (Random Forest), "xgb" (XGBoost), and "knn" (K-Nearest Neighbors).

        Raises
        ------
        ValueError
            If the model configuration includes classifiers but not regressors, an error is raised 
            since classifiers are not supported for parameter merging.

        Notes
        -----
        The method processes the parameters of each model for regression and ensemble models, 
        extracting hyperparameters such as `n_estimators`, `max_depth`, `learning_rate`, and others.
        The parameters for each target are aggregated into a DataFrame and saved as a CSV file in the 
        "posts/parameters" directory.
        """
        
        all_parameters = []

        for i in range(len(self.unique_targets)):
            
            target = self.unique_targets[i]
            print("the target is:")
            print(target)
            target_no_space = target.replace(' ', '_')

            with open(os.path.join(self.root, self.model_config['path_out'], self.model_config['run_name'], "model", model, target_no_space) + self.extension, 'rb') as file:

                m = pickle.load(file)

            if self.model_type == "reg":

                if model == "rf":
                    max_depth = m.regressor_.named_steps.estimator.max_depth
                    max_features = m.regressor_.named_steps.estimator.max_features
                    max_samples = m.regressor_.named_steps.estimator.max_samples
                    min_samples_leaf = m.regressor_.named_steps.estimator.min_samples_leaf
                    n_estimators = m.regressor_.named_steps.estimator.n_estimators
                    parameters = pd.DataFrame({'target':[target], 'n_estimators':[n_estimators], 'max_features':[max_features], 'max_depth':[max_depth], 
                                            'min_samples_leaf':[min_samples_leaf], 'max_samples':[max_samples]
                                            })
                    all_parameters.append(parameters)
                elif model == "xgb":
                    learning_rate = m.regressor_.named_steps.estimator.learning_rate
                    n_estimators = m.regressor_.named_steps.estimator.n_estimators
                    max_depth = m.regressor_.named_steps.estimator.max_depth
                    subsample = m.regressor_.named_steps.estimator.subsample
                    colsample_bytree = m.regressor_.named_steps.estimator.colsample_bytree
                    gamma = m.regressor_.named_steps.estimator.gamma
                    alpha = m.regressor_.named_steps.estimator.reg_alpha
                    parameters = pd.DataFrame({'target':[target], 'learning_rate':[learning_rate], 'n_estimators':[n_estimators], 
                                            'max_depth':[max_depth], 'subsample':[subsample], 'colsample_bytree':[colsample_bytree],
                                            'learning_rate':[learning_rate], 'gamma':[gamma], 'alpha':[alpha]                                           
                                            })
                    all_parameters.append(parameters)
                elif model == "knn":
                    max_samples = m.regressor_.named_steps.estimator.max_samples
                    max_features = m.regressor_.named_steps.estimator.max_features
                    leaf_size = m.regressor_.named_steps.estimator.estimator.leaf_size
                    n_neighbors = m.regressor_.named_steps.estimator.estimator.n_neighbors
                    p = m.regressor_.named_steps.estimator.estimator.p
                    weights = m.regressor_.named_steps.estimator.estimator.weights
                    parameters = pd.DataFrame({'target':[target], 'max_samples':[max_samples], 'max_features':[max_features],
                                            'leaf_size':[leaf_size], 'n_neighbors':[n_neighbors], 'p':[p], 'weights':[weights]
                                            })
                    all_parameters.append(parameters) 

            elif self.model_type == "clf":
                raise ValueError("classifiers are not supported")

            elif self.model_type == "zir":
                if model == "rf":
                    max_depth_reg = m.regressor_.regressor.named_steps.estimator.max_depth
                    max_features_reg = m.regressor_.regressor.named_steps.estimator.max_features
                    max_samples_reg = m.regressor_.regressor.named_steps.estimator.max_samples
                    min_samples_leaf_reg = m.regressor_.regressor.named_steps.estimator.min_samples_leaf
                    n_estimators_reg = m.regressor_.regressor.named_steps.estimator.n_estimators

                    n_estimators_clf = m.classifier.named_steps.estimator.n_estimators
                    max_features_clf = m.classifier.named_steps.estimator.max_features
                    max_depth_clf = m.classifier.named_steps.estimator.max_depth
                    min_samples_leaf_clf = m.classifier.named_steps.estimator.min_samples_leaf
                    max_samples_clf = m.classifier.named_steps.estimator.max_samples

                    parameters = pd.DataFrame({'target':[target], 'reg_n_estimators':[n_estimators_reg], 
                                            'reg_max_features':[max_features_reg], 'reg_max_depth':[max_depth_reg], 
                                            'reg_min_samples_leaf':[min_samples_leaf_reg], 'reg_max_samples':[max_samples_reg],
                                            'clf_n_estimators':[n_estimators_clf], 
                                            'clf_max_features':[max_features_clf], 'clf_max_depth':[max_depth_clf], 
                                            'clf_min_samples_leaf':[min_samples_leaf_clf], 'clf_max_samples':[max_samples_clf]
                                            })
                    all_parameters.append(parameters)

                elif model == "xgb":
                    learning_rate_reg = m.regressor_.regressor.named_steps.estimator.learning_rate
                    n_estimators_reg = m.regressor_.regressor.named_steps.estimator.n_estimators
                    max_depth_reg = m.regressor_.regressor.named_steps.estimator.max_depth
                    subsample_reg = m.regressor_.regressor.named_steps.estimator.subsample
                    colsample_bytree_reg = m.regressor_.regressor.named_steps.estimator.colsample_bytree
                    gamma_reg = m.regressor_.regressor.named_steps.estimator.gamma
                    alpha_reg = m.regressor_.regressor.named_steps.estimator.reg_alpha

                    learning_rate_clf = m.classifier.named_steps.estimator.learning_rate
                    n_estimators_clf = m.classifier.named_steps.estimator.n_estimators
                    max_depth_clf = m.classifier.named_steps.estimator.max_depth
                    subsample_clf = m.classifier.named_steps.estimator.subsample
                    colsample_bytree_clf = m.classifier.named_steps.estimator.colsample_bytree
                    gamma_clf = m.classifier.named_steps.estimator.gamma
                    alpha_clf = m.classifier.named_steps.estimator.reg_alpha


                    parameters = pd.DataFrame({'target':[target], 'reg_learning_rate':[learning_rate_reg], 'reg_n_estimators':[n_estimators_reg], 
                                            'reg_max_depth':[max_depth_reg], 'reg_subsample':[subsample_reg], 'reg_colsample_bytree':[colsample_bytree_reg],
                                            'reg_learning_rate':[learning_rate_reg], 'reg_gamma':[gamma_reg], 'reg_alpha':[alpha_reg],
                                            'clf_learning_rate':[learning_rate_clf], 'clf_n_estimators':[n_estimators_clf], 
                                            'clf_max_depth':[max_depth_clf], 'clf_subsample':[subsample_clf], 'clf_colsample_bytree':[colsample_bytree_clf],
                                            'clf_learning_rate':[learning_rate_clf], 'clf_gamma':[gamma_clf], 'clf_alpha':[alpha_clf]                                           
                                            })
                    all_parameters.append(parameters)

                elif model == "knn":
                    max_samples_reg = m.regressor_.regressor.named_steps.estimator.max_samples
                    max_features_reg = m.regressor_.regressor.named_steps.estimator.max_features
                    leaf_size_reg = m.regressor_.regressor.named_steps.estimator.estimator.leaf_size
                    n_neighbors_reg = m.regressor_.regressor.named_steps.estimator.estimator.n_neighbors
                    p_reg = m.regressor_.regressor.named_steps.estimator.estimator.p
                    weights_reg = m.regressor_.regressor.named_steps.estimator.estimator.weights

                    max_samples_clf = m.classifier.named_steps.estimator.max_samples
                    max_features_clf = m.classifier.named_steps.estimator.max_features
                    leaf_size_clf = m.classifier.named_steps.estimator.estimator.leaf_size
                    n_neighbors_clf = m.classifier.named_steps.estimator.estimator.n_neighbors
                    p_clf = m.classifier.named_steps.estimator.estimator.p
                    weights_clf = m.classifier.named_steps.estimator.estimator.weights



                    parameters = pd.DataFrame({'target':[target], 'reg_max_samples':[max_samples_reg], 'reg_max_features':[max_features_reg],
                                            'reg_leaf_size':[leaf_size_reg], 'reg_n_neighbors':[n_neighbors_reg], 
                                            'reg_p':[p_reg], 'reg_weights':[weights_reg],
                                            'clf_max_samples':[max_samples_clf], 'clf_max_features':[max_features_clf],
                                            'clf_leaf_size':[leaf_size_clf], 'clf_n_neighbors':[n_neighbors_clf], 
                                            'clf_p':[p_clf], 'clf_weights':[weights_clf]
                                            })
                    all_parameters.append(parameters) 

        all_parameters= pd.concat(all_parameters)
        try: #make new dir if needed
            os.makedirs(os.path.join(self.root, self.model_config['path_out'], self.model_config['run_name'], "posts/parameters"))
        except:
            None
        all_parameters.to_csv(os.path.join(self.root, self.model_config['path_out'], self.model_config['run_name'], "posts/parameters", model) + "_parameters.csv", index=False)

        
        print("finished merging parameters")

    def estimate_carbon(self, variable):

        """
        Estimate carbon content for each target based on a specified variable.

        This method calculates the carbon content for each target by scaling the data in `self.d` 
        with the values of the specified variable from the `traits` DataFrame. The results are 
        stored back in `self.d`.

        Parameters
        ----------
        variable : str
            The name of the column in the `traits` DataFrame containing the carbon content values 
            to be used for scaling the target data.
        """

        w = self.traits.query('Target in @self.targets')
        var = w[variable].to_numpy()
        print(var)
        self.d = self.d.apply(lambda row : (row[self.targets]* var), axis = 1)
        print("finished estimating " + variable)

    def def_groups(self, dict):
        """
        Define groups of species based on a provided dictionary.

        Parameters
        ----------
        dict : dict
            A dictionary where keys represent group names, and values are lists of species or 
            column names to be grouped under each key.
            
        Notes
        -----
        - The method renames columns in `self.d` based on the provided dictionary and then sums 
        their values to create grouped columns.
        - The resulting grouped data is concatenated to the original `self.d`.
        """     

        df = self.d[self.targets]
        df = (df.rename(columns=dict)
            .groupby(level=0, axis=1, dropna=False)).sum( min_count=1)
        self.d = pd.concat([self.d, df], axis=1)
        print("finished defining groups")

    def cwm(self, variable):
        """
        Calculate community weighted mean values for a given parameter. 

        Parameters
        ----------

        variable : string
            variable that is used to estimate cwm.

        """

        w = self.traits.query('Target in @self.targets')
        var = w[variable].to_numpy()
        var_name = 'cwm ' + variable
        self.d[var_name] = self.d.apply(lambda row : np.average(var, weights=row[self.targets]), axis = 1)
        print("finished calculating CWM " + variable)

    def diversity(self):
        """
        Estimates Shannon diversity using scikit-bio.
        """
        self.d['shannon'] = self.d.apply(shannon, axis=1)
        print("finished calculating shannon diversity")

    def total(self):
        """
        Sum target rows to estimate total.

        Notes
        ----------
        Useful for estimating total species abundances or varable sum if targets are continuous.
        Total is estimated based on the target list defined in model_config. 

        """

        self.d['total'] = self.d[self.targets].sum( axis='columns')
        self.d['total_log'] = np.log(self.d['total'])
        print("finished calculating total")

    def integration(self, *args, **kwargs):
        return self.integration_class(self, *args, **kwargs)

    class integration:
        def __init__(self, parent, 
                     resolution_lat=1.0, resolution_lon=1.0, depth_w=5, 
                     vol_conversion=1, magnitude_conversion=1, molar_mass=1, rate=False):
            """
            Parameters
            ----------
            resolution_lat : float
                Latitude resolution in degrees, default is 1.0 degree.
            
            resolution_lon : float
                Longitude resolution in degrees, default is 1.0 degree.
            
            depth_w : float
                Bin depth in meters, default is 5m.

            vol_conversion : float
                Conversion to m^3, e.g., l to m^3 would be 1e3, default is 1 (no conversion).
            
            magnitude_conversion : float
                Prefix conversion, e.g., umol to Pmol would be 1e-21, default is 1 (no conversion).
            
            molar_mass : float
                Conversion from mol to grams, default is 1 (no conversion). Optional: 12.01 (carbon).
            
            rate : bool
                If input data is in rate per day, integrates over each month to provide an annual rate (yr^-1).            
            """


            self.parent = parent
            self.resolution_lat = resolution_lat
            self.resolution_lon = resolution_lon
            self.depth_w = depth_w
            self.vol_conversion = vol_conversion
            self.magnitude_conversion = magnitude_conversion
            self.molar_mass = molar_mass
            self.rate = rate
            self.calculate_volume()

        def calculate_volume(self):
            """
            Calculate the volume for each cell and add it as a new field to the dataset.

            Examples
            --------
            >>> m = post(model_config)
            >>> int = m.Integration(m, resolution_lat=1.0, resolution_lon=1.0, depth_w=5, vol_conversion=1, magnitude_conversion=1e-21, molar_mass=12.01, rate=True)
            >>> print("Volume calculated:", int.ds['volume'].values)

            """            
            ds = self.parent.d.to_xarray()
            resolution_lat = self.resolution_lat
            resolution_lon = self.resolution_lon
            depth_w = self.depth_w

            # Calculate the number of cells in latitude and longitude
            num_cells_lat = int(ds['lat'].size / resolution_lat)
            num_cells_lon = int(ds['lon'].size / resolution_lon)

            # Retrieve initial latitude and longitude bound
            min_lat = ds['lat'].values[0]
            min_lon = ds['lon'].values[0]

            # Initialize the 2D array to store the areas
            area = np.zeros((num_cells_lat, num_cells_lon))

            earth_radius = 6371000.0  # Earth's radius in meters

            # Calculate the area of each cell
            for lat_index in range(num_cells_lat):
                for lon_index in range(num_cells_lon):
                    # Calculate the latitude range of the cell
                    lat_bottom = min_lat + lat_index * resolution_lat
                    lat_top = lat_bottom + resolution_lat

                    # Calculate the longitude range of the cell
                    lon_left = min_lon + lon_index * resolution_lon
                    lon_right = lon_left + resolution_lon

                    # Calculate the area of the grid cell
                    areas = earth_radius ** 2 * (np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bottom))) * \
                            (np.radians(lon_right) - np.radians(lon_left))

                    # Store the area in the array
                    area[lat_index, lon_index] = areas

            volume = area * depth_w
            ds['volume'] = (('lat', 'lon'), volume)
            self.parent.d = ds.to_dataframe()
        
        def integrate_total(self, variable='total', monthly=False, subset_depth=None):
            """
            Estimates global integrated values for a single target. Returns the depth integrated annual total.
            
            Parameters
            ----------
            variable : str
                The field to be integrated. Default is 'total' from PIC or POC Abil output.

            monthly : bool
                Whether or not to calculate a monthly average value instead of an annual total. Default is False.
 
            subset_depth : float
                Depth in meters from surface to which integral should be calculated. Default is None. Ex. 100 for top 100m integral.

            Examples
            --------
            >>> m = post(model_config)
            >>> int = m.Integration(m, resolution_lat=1.0, resolution_lon=1.0, depth_w=5, vol_conversion=1, magnitude_conversion=1e-21, molar_mass=12.01, rate=True)
            >>> result = integration.integrate_total(variable='Calcification')
            >>> print("Final integrated total:", result.values)
            """
            print("Initiate integrated_total")
            ds = self.parent.d.to_xarray()
            vol_conversion = self.vol_conversion
            magnitude_conversion = self.magnitude_conversion
            molar_mass = self.molar_mass
            rate = self.rate

            # Average number of days for each month (accounting for leap years)
            days_per_month_full = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
            
            # Get the available time points (months) from the dataset
            available_time = ds['time'].values  # Assuming 'time' is an array of 1, 2, 3, and 12

            # Subset the days_per_month array to only include the available months
            days_per_month = days_per_month_full[available_time - 1]

            if subset_depth:
                ds = ds.sel(depth=slice(0, subset_depth))

            if rate:
                if monthly:
                    # Calculate monthly total (separately for each month)
                    total = []
                    for i,month in enumerate(available_time):
                        monthly_total = (ds[variable].isel(time=i) * ds['volume'].isel(time=i) * days_per_month[i]).sum(dim=['lat', 'lon', 'depth'])
                        monthly_total = (monthly_total * molar_mass) * vol_conversion * magnitude_conversion
                        total.append(monthly_total)
                    total = xr.concat(total, dim="month")
                    print(f"All monthly totals: {total.values}")
                else:
                    # Calculate annual total
                    total = (ds[variable] * ds['volume'] * days_per_month.mean()).sum(dim=['lat', 'lon', 'depth', 'time'])
                    total = (total * molar_mass) * vol_conversion * magnitude_conversion
                    print("Final integrated total:", total.values)
            else:
                if monthly:
                    # Calculate monthly total (separately for each month)
                    total = []
                    for i,month in enumerate(available_time):
                        monthly_total = (ds[variable].isel(time=i) * ds['volume']).isel(time=i).sum(dim=['lat', 'lon', 'depth'])
                        monthly_total = (monthly_total * molar_mass) * vol_conversion * magnitude_conversion
                        total.append(monthly_total)
                    total = xr.concat(total, dim="month")
                    print(f"All monthly totals: {total.values}")
                else:
                    # Calculate annual total
                    total = (ds[variable] * ds['volume']).sum(dim=['lat', 'lon', 'depth', 'time'])
                    total = (total * molar_mass) * vol_conversion * magnitude_conversion
                    print("Final integrated total:", total.values)
            return total


        def integrated_totals(self, targets=None, monthly=False, subset_depth=None, 
                             export=True, model="ens"):
            """
            Estimates global integrated values for all targets.
    
            Considers latitude and depth bin size.
    
            Parameters
            ----------
            targets : str
                The fields to be integrated. Default is 'total' from PIC or POC Abil output.

            monthly : bool
                Whether or not to calculate a monthly average value instead of an annual total. Default is False.
 
            subset_depth : float
                Depth in meters from surface to which integral should be calculated. Default is None. Ex. 100 for top 100m integral.

            export : bool
                Whether of not to export integrated totals as .csv. Default is True.

            model : str
                The model version to be integrated. Default is "ens". Other options include {"rf", "xgb", "knn"}.
    
            """
            ds = self.parent.d.to_xarray()
            if targets.all == None:
                targets = self.targets
            if "total" in ds:
                targets = np.append(targets, 'total')
            totals = []

            for target in targets:
                try:
                    print(f"Processing target: {target}")
                    total = self.integrate_total(variable=target, monthly=monthly, subset_depth=subset_depth)
                    total_df = pd.DataFrame({'total': [total.values], 'variable': target})
                    totals.append(total_df)
                except Exception as e:
                    print(f"Some targets do not have predictions! Missing: {target}")
                    print(f"Error: {e}")
            totals = pd.concat(totals)

            if export:
                depth_str = f"_depth_{subset_depth}m" if subset_depth else ""
                month_str = "_monthly_int" if monthly else ""
                try: #make new dir if needed
                    os.makedirs(os.path.join(self.parent.root, self.parent.model_config['path_out'], self.parent.model_config['run_name'], "posts/integrated_totals"))
                except:
                    None

                path_out = self.parent.model_config['path_out']
                run_name = self.parent.model_config['run_name']
                #pi = self.parent.pi
                statistic = self.parent.statistic
                datatype = self.parent.datatype

                # Build the full file path
                output_dir = os.path.join(self.parent.root, path_out, run_name, "posts/integrated_totals")
                filename = f"{model}_integrated_totals_{statistic}{depth_str}{month_str}{datatype}.csv"
                file_path = os.path.join(output_dir, filename)

                # Write to CSV
                totals.to_csv(file_path, index=False)

                print(f"Exported totals")

    def estimate_applicability(self):
        """
        Estimate the area of applicability for the data using a strategy similar to Meyer & Pebesma 2022).

        This calculates the importance-weighted feature distances from test to train points,
        and then defines the "applicable" test sites as those closer than some threshold
        distance.
        """

        # create empty dataframe with the same index as X_predict
        aoa_dataset = pd.DataFrame(index=self.X_predict.index)

        # estimate the aoa for each target:
        for i in range(len(self.targets)):
            
            target = self.targets[i]
            target_no_space = target.replace(' ', '_')

            # load the voting regressor model object for each target:
            with open(os.path.join(self.root, self.model_config['path_out'], self.model_config['run_name'], "model", "ens", target_no_space) + self.extension, 'rb') as file:
                m = pickle.load(file)
            
            aoa = area_of_applicability(
                X_test=self.X_predict,
                X_train=self.X_train,
                y_train= self.y_train,
                model=m
            )

            # update the dataframe, where each column name is the target analyzed
            aoa_dataset[target] = aoa

        # convert df to xarray ds:
        aoa_dataset = aoa_dataset.to_xarray()
        
        # add metadata:
        aoa_dataset['lat'].attrs['units'] = 'degrees_north'
        aoa_dataset['lat'].attrs['long_name'] = 'latitude'

        aoa_dataset['lon'].attrs['units'] = 'degrees_east'
        aoa_dataset['lon'].attrs['long_name'] = 'longitude'

        aoa_dataset['depth'].attrs['units'] = 'm'
        aoa_dataset['depth'].attrs['positive'] = 'down'
        
        # export aoa to netcdf:
        aoa_dataset.to_netcdf(os.path.join(self.path_out, "aoa.nc"))


    def merge_env(self):
        """
        Merge model output with environmental data.

        This method aligns and merges the predicted values (model output) with the existing 
        environmental dataset stored in `self.d`. The merged data replaces `self.d`.

        Returns
        -------
        None
        """

        X_predict = self.X_predict.to_xarray()
        ds = self.d.to_xarray()
        aligned_datasets = xr.align(ds,X_predict, join="inner")
        ds = xr.merge(aligned_datasets)
        if 'FID' in ds:
            ds['FID'] = ds['FID'].where(ds['FID'] != '', np.nan)
        self.d = ds.to_dataframe()
        self.d = self.d.dropna()

    def export_ds(self, file_name, 
                  author=None, description=None):
        """
        Export the processed dataset to a NetCDF file.

        This method saves the processed dataset (`self.d`) to a NetCDF file in the location 
        defined by `self.path_out`, with optional metadata such as author and description.

        Parameters
        ----------
        file_name : str 
            The name of the NetCDF file (without extension). 
        author : str, optional
            The name of the author to include in NetCDF metadata (default is None).
        description : str, optional
            A description or title to include in the NetCDF metadata (default is None).

        Notes
        -----
        - The export location is defined in the `model_config.yml` file and is stored in `self.path_out`.
        - The method sets metadata attributes such as conventions, creator name, and units for 
        latitude, longitude, and depth.
        - Missing directories in the export path are created if necessary.
        - The file is saved with a suffix that includes the `pi` value (e.g., `_PI50.nc`).
        """
    
        try: #make new dir if needed
            os.makedirs(self.path_out)
        except:
            None

        print("export_ds")
        print("dataframe: ")
        print(self.d.head())
        ds = self.d.to_xarray()

        if description is not None:
            ds.attrs['description'] = description
        ds.attrs['Conventions'] = 'CF-1.5'
        if author is not None:
            ds.attrs['creator_name'] = author
        try:
            ds['lat'].attrs['units'] = 'degrees_north'
            ds['lat'].attrs['long_name'] = 'latitude'
        except:
            pass
        try:
            ds['lon'].attrs['units'] = 'degrees_east'
            ds['lon'].attrs['long_name'] = 'longitude'
        except:
            pass
        try:
            ds['depth'].attrs['units'] = 'm'
            ds['depth'].attrs['positive'] = 'down'
        except:
            pass
        #to add loop defining units of variables

        print(self.d.head())
        ds.to_netcdf(os.path.join(self.path_out, file_name) + "_" + self.statistic + self.datatype + ".nc")

        print("exported ds to: " + self.path_out + file_name + "_" + self.statistic + self.datatype +  ".nc")
        #add nice metadata


    def export_csv(self, file_name):
        """
        Export the processed dataset to a csv file.

        This method saves the processed dataset (`self.d`) to a csv file in the location 
        defined by `self.path_out`, with optional metadata such as author and description.

        Parameters
        ----------
        file_name : str 
            The name of the csv file (without extension). 
        
        Notes
        -----
        - The export location is defined in the `model_config.yml` file and is stored in `self.path_out`.
        - Missing directories in the export path are created if necessary.
        - The file is saved with a suffix that includes the `pi` value (e.g., `_PI50.nc`).
        """
    
        try: #make new dir if needed
            os.makedirs(self.path_out)
        except:
            None
    
        print(self.d.head())
        self.d.to_csv(os.path.join(self.path_out, file_name) + "_" + self.statistic + self.datatype + ".csv")

        print("exported d to: " + self.path_out + file_name + "_" + self.statistic + self.datatype + ".csv")
        #add nice metadata

    def merge_obs(self, file_name, targets=None):
        """
        Merge model output with observational data and calculate residuals.

        This function integrates model predictions with observational data based on 
        spatial and temporal indices, calculates residuals, and exports the merged dataset.

        Parameters
        ----------
        file_name : str
            The base name of the output file to save the merged dataset.
        targets : list of str, optional
            A list of target variable names to include in the merge. If None, the default 
            targets from `self.targets` are used (default is None).

        Notes
        -----
        - The function matches the observational data with model predictions based on the 
        indices `['lat', 'lon', 'depth', 'time']`.
        - Residuals are calculated as `observed - predicted` for each target variable.
        - Columns included in the output are the original targets, their modeled values 
        (suffixed with `_mod`), and their residuals (suffixed with `_resid`).
        - The merged dataset is saved as a CSV file with a suffix `_PI` followed by the 
        `pi` value, appended to the output file name.
        - Observational data is loaded from the path defined in `self.model_config['training']`.

        Raises
        ------
        FileNotFoundError
            If the observational dataset file cannot be found at the specified location.
        """
        # Select and rename the target columns for d
        if targets.all == None:
            targets = self.targets
        d = self.d[targets]

        mod_columns = {target: target + '_mod' for target in targets}
        d = d.rename(mod_columns, axis=1)
        d.reset_index(inplace=True)
        d.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)        

        # Read the training targets from the training.csv file defined in model_config
        try:
            df2_path = os.path.join(self.root, self.model_config['training'])

            df2 = pd.read_csv(df2_path)
        except:
            raise FileNotFoundError(f"Dataset not found at {df2_path}")
        

        df2.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)
        df2['dummy'] = 1

        out = pd.concat([df2, d], axis=1)
        out = out[out['dummy'] == 1].drop(['dummy'], axis=1)

        # Calculate residuals
        for target in targets:
            out[target + '_resid'] = out[target] - out[target + '_mod']

        # Define the columns to keep in the final DataFrame
        keep_columns = list(targets) + list(mod_columns.values()) + [target + '_resid' for target in targets]

        out = out[keep_columns]
        file_name = f"{file_name}_obs"
        print(out.head())
        out.to_csv(os.path.join(self.path_out, file_name)  + self.datatype +  ".csv")

        print("exported d to: " + self.path_out + file_name  + self.datatype + ".csv")

        print('training merged with predictions')