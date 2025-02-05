import pandas as pd
import numpy as np
import glob, os
import xarray as xr
import pickle
import gc
from yaml import dump, Dumper
from skbio.diversity.alpha import shannon

class post:
    """
    Post-process results of ensemble.

    This class handles tasks such as merging NetCDF files, loading target data, 
    and organizing model results for further analysis.

    Attributes
    ----------
    path_out : str
        The output directory path where the model results are saved.
    ds : xarray.Dataset
        The dataset containing the merged data from NetCDF files.
    targets : pd.DataFrame
        A DataFrame containing target trait information loaded from a CSV file.
    root : str
        The root directory specified in the model configuration.
    d : pd.DataFrame
        A DataFrame representation of the dataset after conversion and cleaning.
    targets : pd.Series
        The target values extracted from the trait data that are present in the dataset columns.
    model_config : dict
        The model configuration dictionary containing paths, parameters, and ensemble configuration.
    pi : str
        Input parameter identifier.
    model_type : str
        The type of model being used, determined from the ensemble configuration ("zir" or "reg").
    extension : str
        File extension used for saving the model (e.g., "_zir.sav").
    datatype : str, optional
        The datatype being processed, appended to data exports (if provided).

    Methods
    -------
    merge_netcdf(path_in)
        Merges multiple NetCDF files from a specified directory into a single xarray.Dataset.

    export_model_config()
        Exports the model configuration details stored in `self.model_config` to a file for documentation or replication.

    merge_performance()
        Merges performance metrics (e.g., RMSE, R²) from multiple models into a single performance summary.

    merge_performance_single_model(model)
        Merges performance metrics for a specific model into the dataset.

    merge_parameter()
        Aggregates and merges parameter values across all models into a single dataset.

    merge_parameters_single_model(model)
        Merges parameter values for a specific model into the dataset.

    estimate_carbon(variable)
        Estimates carbon-based metrics using the specified variable. Calculates derived metrics from carbon estimates.

    def_groups(dict)
        Defines functional or taxonomic groups for grouping analyses, based on the provided dictionary.

    cwm(variable)
        Calculates the community-weighted mean (CWM) for the specified variable.

    diversity()
        Calculates the Shannon diversity index for the dataset.

    total()
        Computes the total value of a specified variable over a specific region, depth, or time period.
    
    integration(*args, **kwargs)
        Initializes an `integration` object to perform data volume calculation and global integrations.

    integration.calculate_volume()
        Calculates the volume of each grid cell in the dataset and adds it as a new field.

    integration.integrate_total(variable='total', monthly=False, subset_depth=None)
        Estimates global integrated values for a single target variable.
        Supports depth integration, monthly averages, and annual totals.

    integration.integrated_totals(targets=None, monthly=False, subset_depth=None, export=True, model="ens")
        Calculates global integrated values for all specified target variables.
        Optionally exports the results as a CSV file.

    merge_obs(file_name, targets=None)
        Merges model output with observational data and calculates residuals.
        Saves the merged dataset as a CSV file.

    export_csv(file_name)
        Exports the processed dataset (`self.d`) to a CSV file in the location defined by `self.path_out`.

    export_ds(file_name, author=None, description=None)
        Exports the processed dataset (`self.d`) to a NetCDF file.
        Includes optional metadata like author and description.

    merge_env(X_predict)
        Aligns and merges the predicted values (`X_predict`) with the existing environmental dataset (`self.d`).
        Replaces `self.d` with the merged dataset.
    """
    def __init__(self, model_config, pi="50", datatype=None):
        """
        Initialize the `post` class with model configuration, input parameters, and optional datatype.

        Parameters
        ----------
        model_config : dict
            A dictionary containing configuration settings for the model, including paths, parameters, and ensemble configuration.
        pi : str, optional, default="50"
            Prediction interval from crossfolds used for model naming. Default is 50 for median prediction.
        datatype : str, optional
            The datatype being processed (e.g., "pg poc"), which is appended to exported data files.
        
        Returns
        -------
        None

        Methods
        -------
        merge_netcdf(path_in):
            Merges multiple NetCDF files from the specified directory into a single dataset.
        """
        def merge_netcdf(path_in):
            """
            Merge multiple NetCDF files from a specified directory into a single dataset.

            This method combines all NetCDF files in the given directory (matching the pattern "*.nc") 
            into one xarray.Dataset using `xarray.open_mfdataset`. It outputs status messages 
            to indicate the start and completion of the process.

            Parameters
            ----------
            path_in : str
                The directory path containing the NetCDF files to be merged.

            Returns
            -------
            xarray.Dataset
                A dataset containing the combined data from all NetCDF files in the directory.

            Notes
            -----
            - The merging process is performed in memory, which might require sufficient RAM depending 
            on the size of the NetCDF files.
            - This method assumes that all NetCDF files in the directory are compatible for merging.
            """
            print("merging...")
            ds = xr.open_mfdataset(os.path.join(path_in, "*.nc"))
            print("finished loading netcdf files")
            return(ds)

        self.path_out = os.path.join(model_config['root'], model_config['path_out'], model_config['run_name'], "posts/")
        self.ds = merge_netcdf(os.path.join(model_config['root'], model_config['path_out'], model_config['run_name'], model_config['path_in'], pi))
        self.traits = pd.read_csv(os.path.join(model_config['root'], model_config['targets']))

        self.root  =  model_config['root'] 

        self.d = self.ds.to_dataframe()
        self.d = self.d.dropna()
        self.targets = self.traits['Target'][self.traits['Target'].isin(self.d.columns.values)]
        self.model_config = model_config
        self.pi = pi

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
        Export the model configuration dictionary to a YAML file.

        This method saves the `model_config` dictionary as a YAML file named "model_config.yml" 
        in the directory specified by `self.path_out`. If the directory does not exist, it is created.

        Raises
        ------
        Exception
            If an error occurs during the directory creation or file writing process, an exception
            is caught and an error message is printed.

        Notes
        -----
        - The YAML file is saved as "model_config.yml" in the `self.path_out` directory.
        - The function ensures the output directory exists before writing the file.
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
        Merge performance data for all models specified in the ensemble configuration.

        This method iterates over all models listed in the `ensemble_config` dictionary, as well as the ensemble model, 
        and merges their performance metrics. The merging process for each model is handled by the 
        `merge_performance_single_model` method.

        Notes
        -----
        - The method gathers all models defined in the ensemble configuration, along with the ensemble model ("ens").
        - For each model, the performance metrics are processed and saved using the `merge_performance_single_model` method.
        """
        models = [value for key, value in self.model_config['ensemble_config'].items() if key.startswith("m")]
        print("models included in merge performance!")
        print(models)
        models.append("ens")
        for model in models:
            self.merge_performance_single_model(model)

       
    def merge_performance_single_model(self, model):
        """
        Merge performance metrics for a single model and save the results to a CSV file.

        Parameters
        ----------
        model : str
            The name of the model for which performance metrics are being merged. Performance data 
            is expected to be stored in a pickle file in the "scoring" directory under the model name 
            and target name.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the model configuration specifies a classifier but not a regressor, an error is raised 
            since classifiers are not supported for performance merging.

        Notes
        -----
        - This method calculates several performance metrics for each target column in the dataset:
            - R2: Coefficient of determination.
            - RMSE: Root Mean Squared Error.
            - MAE: Mean Absolute Error.
            - rRMSE: Relative Root Mean Squared Error (normalized by mean).
            - rMAE: Relative Mean Absolute Error (normalized by mean).
        - Performance metrics are aggregated into a DataFrame and saved as a CSV file in the 
        "posts/performance" directory under the specified model name.
        - If the directory does not exist, it is created.
        """
        all_performance = []

        for i in range(len(self.d.columns)):
            target = self.d.columns[i]
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
        Merges model parameters for multiple models as specified in the ensemble configuration.

        Notes
        -----
        - The method iterates over each model specified in the `ensemble_config` of the `model_config`.
        - For each model, it calls `merge_parameters_single_model` to extract and merge hyperparameters.
        - The merged parameters are saved as CSV files in the "posts/parameters" directory for each model.
        """

        models = [value for key, value in self.model_config['ensemble_config'].items() if key.startswith("m")]
        for model in models:
            self.merge_parameters_single_model(model)

    def merge_parameters_single_model(self, model):
        """
        Merges and saves hyperparameters for a single model.

        Parameters
        ----------
        model : str
            The name of the model for which parameters are being merged. Examples include "rf" (Random Forest),
            "xgb" (XGBoost), or "knn" (K-Nearest Neighbors). Model must be one off.

        Raises
        ------
        ValueError
            If the model type is set to "clf" (classifier), as classifiers are not supported for this operation.

        Notes
        -----
        - The function extracts hyperparameters such as `n_estimators`, `max_depth`, and others, based on the model type.
        - It supports regression ("reg"), ensemble ("zir"), and classifier ("clf") model types, but only regression
          and ensemble models are processed.
        - For ensemble models ("zir"), both regression and classification hyperparameters are extracted.
        - The aggregated parameters for all target columns are saved as a CSV file in the "posts/parameters" directory.
        """
        
        all_parameters = []

        for i in range(len(self.d.columns)):
            
            target = self.d.columns[i]
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
        Estimate the carbon content for each target using a specified trait variable.

        Parameters
        ----------
        variable : str
            The column name in the `traits` DataFrame containing the scaling values to estimate
            carbon content for the targets.

        Notes
        -----
        - The method multiplies the values in the target columns of `self.d` by the corresponding values
          of the specified variable from the `traits` DataFrame.
        - The scaled values are saved back to `self.d`.
        """
        w = self.traits.query('Target in @self.targets')
        var = w[variable].to_numpy()
        print(var)
        self.d = self.d.apply(lambda row : (row[self.targets]* var), axis = 1)
        print("finished estimating " + variable)

    def def_groups(self, dict):
        """
        Define groups of species or targets based on a provided mapping.

        Parameters
        ----------
        dict : dict
            A dictionary where keys are group names and values are lists of column names (species or targets)
            to be grouped under each key.

        Notes
        -----
        - The method renames the columns of `self.d` based on the provided dictionary.
        - It sums the grouped columns to create new columns for each group.
        - The new grouped data is concatenated with the original `self.d`.
        """    

        df = self.d[self.targets]
        df = (df.rename(columns=dict)
            .groupby(level=0, axis=1, dropna=False)).sum( min_count=1)
        self.d = pd.concat([self.d, df], axis=1)
        print("finished defining groups")

    def cwm(self, variable):
        """
        Calculate the community weighted mean (CWM) for a specified trait variable.

        Parameters
        ----------
        variable : str
            The name of the column in the `traits` DataFrame used to compute the CWM.

        Notes
        -----
        - The CWM is calculated as the weighted mean of the trait variable, with weights derived from
          the values in the target columns of `self.d`.
        - The calculated CWM is added as a new column in `self.d` with the name "cwm <variable>".
        """

        w = self.traits.query('Target in @self.targets')
        var = w[variable].to_numpy()
        var_name = 'cwm ' + variable
        self.d[var_name] = self.d.apply(lambda row : np.average(var, weights=row[self.targets]), axis = 1)
        print("finished calculating CWM " + variable)

    def diversity(self):
        """
        Calculate Shannon diversity for the targets.

        Notes
        -----
        - Shannon diversity is calculated using the `shannon` function, which operates on the target columns of `self.d`.
        - The resulting values are stored in a new column "shannon" in `self.d`.
        """
        self.d['shannon'] = self.d.apply(shannon, axis=1)
        print("finished calculating shannon diversity")

    def total(self):
        """
        Calculate the total and logarithmic total of target values.

        Notes
        -----
        - The total is computed by summing the values across all target columns defined in the `self.targets` list.
        - The logarithm of the total is also calculated and saved as a separate column "total_log".
        - This is useful for estimating total species abundance or other continuous target sums.
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
            Initialize the `integration` class with parameters for spatial and volumetric calculations.

            Parameters
            ----------
            parent : object
                The parent class instance containing the dataset and configuration.

            resolution_lat : float, optional
                Latitude resolution in degrees. Default is 1.0.

            resolution_lon : float, optional
                Longitude resolution in degrees. Default is 1.0.

            depth_w : float, optional
                Bin depth in meters. Default is 5.

            vol_conversion : float, optional
                Conversion factor for volume, e.g., from liters to cubic meters (1e3). Default is 1 (no conversion).

            magnitude_conversion : float, optional
                Conversion factor for magnitude, e.g., from micromoles to petamoles (1e-21). Default is 1 (no conversion).

            molar_mass : float, optional
                Conversion from moles to grams. Default is 1 (no conversion). Example: 12.01 for carbon.

            rate : bool, optional
                Whether to integrate over each month to provide an annual rate (yr^-1) for rate-based data. Default is False.
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
            Calculate the volume for each spatial cell based on latitude, longitude, and depth resolution.

            Notes
            -----
            - Calculates the area for each latitude and longitude cell using Earth's radius and trigonometric formulas.
            - Multiplies the area by the depth to compute the volume for each cell.
            - Adds the computed volume as a new field in the dataset.

            Examples
            --------
            >>> m = post(model_config)
            >>> integration = m.integration(m, resolution_lat=1.0, resolution_lon=1.0, depth_w=5)
            >>> print("Volume calculated:", integration.parent.d['volume'].values)
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
            Integrate global values for a single target variable over depth and time.

            Parameters
            ----------
            variable : str, optional
                The field to be integrated. Default is 'total'.

            monthly : bool, optional
                Whether to calculate a monthly average instead of an annual total. Default is False.

            subset_depth : float, optional
                Maximum depth in meters for integration. Default is None (integrate over all depths).

            Returns
            -------
            xarray.DataArray
                Integrated total values, either as an annual or monthly series.

            Examples
            --------
            >>> m = post(model_config)
            >>> integration = m.integration(m, resolution_lat=1.0, resolution_lon=1.0, depth_w=5, rate=True)
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
            Estimate global integrated values for all target variables.

            Parameters
            ----------
            targets : list of str, optional
                The list of target variables to integrate. Default includes all targets in the dataset.

            monthly : bool, optional
                Whether to calculate monthly averages instead of annual totals. Default is False.

            subset_depth : float, optional
                Maximum depth in meters for integration. Default is None (integrate over all depths).

            export : bool, optional
                Whether to export the integrated totals to a CSV file. Default is True.

            model : str, optional
                The model version to integrate. Default is "ens". Other options include {"rf", "xgb", "knn"}.

            Returns
            -------
            pandas.DataFrame
                A DataFrame containing the integrated totals for each target variable.

            Notes
            -----
            - The method iterates through each target and calculates the integrated total.
            - If `export` is True, the results are saved as a CSV file in the appropriate output directory.

            Examples
            --------
            >>> m = post(model_config)
            >>> integ = m.integration(m, resolution_lat=1.0, resolution_lon=1.0, depth_w=5)
            >>> integ.integrated_totals(targets, subset_depth=100)
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
                pi = self.parent.pi
                datatype = self.parent.datatype

                # Build the full file path
                output_dir = os.path.join(self.parent.root, path_out, run_name, "posts/integrated_totals")
                filename = f"{model}_integrated_totals_PI{pi}{depth_str}{month_str}{datatype}.csv"
                file_path = os.path.join(output_dir, filename)

                # Write to CSV
                totals.to_csv(file_path, index=False)

                print(f"Exported totals")


    def merge_env(self, X_predict):
        """
        Merge model output with environmental data.

        This method aligns and merges the predicted values (`X_predict`) with the existing 
        environmental dataset (`self.d`). The merged dataset replaces `self.d`.

        Parameters
        ----------
        X_predict : pd.DataFrame
            A DataFrame containing the model's predicted values to be merged with the 
            environmental dataset.

        Notes
        -----
        - Uses `xarray.align` with `join="inner"` to align the datasets based on their shared dimensions.
        - Adds the merged dataset to `self.d` after dropping any rows with missing values.
        - Ensures the 'FID' field does not contain empty strings.

        Examples
        --------
        >>> m = post(model_config)
        >>> X_predict = pd.DataFrame(...)  # Model predictions
        >>> m.merge_env(X_predict)
        """

        X_predict = X_predict.to_xarray()
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
            The name of the author to include in the NetCDF metadata. Default is None.

        description : str, optional
            A description or title to include in the NetCDF metadata. Default is None.

        Notes
        -----
        - The export path is defined in `self.path_out`, and directories are created if necessary.
        - Adds metadata attributes such as `Conventions`, `creator_name`, and units for latitude, longitude, and depth.
        - Appends a suffix `_PI<pi>.nc` to the output file name, where `<pi>` is the `self.pi` value.

        Examples
        --------
        >>> file_name = model_config['run_name']
        >>> m = post(model_config)
        >>> m.export_ds(file_name, author="Author Name", description="Processed dataset.")
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

        ds['lat'].attrs['units'] = 'degrees_north'
        ds['lat'].attrs['long_name'] = 'latitude'

        ds['lon'].attrs['units'] = 'degrees_east'
        ds['lon'].attrs['long_name'] = 'longitude'

        ds['depth'].attrs['units'] = 'm'
        ds['depth'].attrs['positive'] = 'down'

        #to add loop defining units of variables

        print(self.d.head())
        ds.to_netcdf(os.path.join(self.path_out, file_name) + "_PI" + self.pi + self.datatype + ".nc")

        print("exported ds to: " + self.path_out + file_name + "_PI" + self.pi + self.datatype +  ".nc")
        
        #add nice metadata


    def export_csv(self, file_name):
        """
        Export the processed dataset to a CSV file.

        This method saves the processed dataset (`self.d`) to a CSV file in the location 
        defined by `self.path_out`.

        Parameters
        ----------
        file_name : str
            The name of the CSV file (without extension).

        Notes
        -----
        - The export path is defined in `self.path_out`, and directories are created if necessary.
        - Appends a suffix `_PI<pi>.csv` to the output file name, where `<pi>` is the `self.pi` value.

        Examples
        --------
        >>> file_name = model_config['run_name']
        >>> m = post(model_config)
        >>> m.export_csv(file_name)
        """
    
        try: #make new dir if needed
            os.makedirs(self.path_out)
        except:
            None
    
        print(self.d.head())
        self.d.to_csv(os.path.join(self.path_out, file_name) + "_PI" + self.pi + self.datatype + ".csv")

        print("exported d to: " + self.path_out + file_name + "_PI" + self.pi + self.datatype + ".csv")
        #add nice metadata

    def merge_obs(self, file_name, targets=None):
        """
        Merge model output with observational data and calculate residuals.

        This method integrates model predictions with observational data based on 
        spatial and temporal indices, calculates residuals, and exports the merged dataset.

        Parameters
        ----------
        file_name : str
            The base name of the output file to save the merged dataset.

        targets : list of str, optional
            A list of target variable names to include in the merge. If None, the default 
            targets from `self.targets` are used. Default is None.

        Notes
        -----
        - Matches observational data with model predictions based on indices: `['lat', 'lon', 'depth', 'time']`.
        - Residuals are calculated as `observed - predicted` for each target variable.
        - Columns in the output include the original targets, their modeled values (suffixed with `_mod`), and residuals (suffixed with `_resid`).
        - Saves the merged dataset as a CSV file with a suffix `_obs_PI<pi>.csv`.
        - Observational data is loaded from the path defined in `self.model_config['training']`.

        Raises
        ------
        FileNotFoundError
            If the observational dataset file cannot be found at the specified location.

        Examples
        --------
        >>> file_name = model_config['run_name']
        >>> targets = pd.read_csv(root + model_config['targets'])
        >>> targets =  targets['Target'].values
        >>> m = post(model_config)
        >>> m.merge_obs(file_name, targets)
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
        out.to_csv(os.path.join(self.path_out, file_name) + "_PI" + self.pi + self.datatype +  ".csv")

        print("exported d to: " + self.path_out + file_name + "_PI" + self.pi + self.datatype + ".csv")

        print('training merged with predictions')