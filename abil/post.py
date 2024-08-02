import pandas as pd
import numpy as np
import glob, os
import xarray as xr
import pickle
import gc
#from skbio.diversity.alpha import shannon

class post:
    """
    Post processing of SDM
    """
    def __init__(self, model_config, pi="50"):

        def merge_netcdf(path_in):
            print("merging...")
            ds = xr.open_mfdataset(os.path.join(path_in, "*.nc"))
            print("finished loading netcdf files")
            return(ds)
        
        if model_config['hpc']==False:
            self.path_out = model_config['local_root'] + model_config['path_out'] + model_config['run_name'] + "/posts/"
            self.ds = merge_netcdf(model_config['local_root'] + model_config['path_out'] + model_config['run_name'] + model_config['path_in'] + pi + "/")
            self.traits = pd.read_csv(model_config['local_root'] + model_config['targets'])
            self.root  =  model_config['local_root'] 

        elif model_config['hpc']==True:
            self.path_out = model_config['hpc_root'] + model_config['path_out'] + model_config['run_name'] + "/posts/"
            self.ds = merge_netcdf(model_config['hpc_root'] + model_config['path_out'] + model_config['run_name'] + model_config['path_in'] + pi + "/")
            self.traits = pd.read_csv(model_config['hpc_root'] + model_config['targets'])
            self.root  =  model_config['hpc_root'] 

        else:
            raise ValueError("hpc True or False not defined in yml")
        self.d = self.ds.to_dataframe()
        self.d = self.d.dropna()
        self.targets = self.traits['Target'][self.traits['Target'].isin(self.d.columns.values)]
        self.model_config = model_config
        self.pi = pi

        # if self.model_config['ensemble_config']['classifier'] and not self.model_config['ensemble_config']['regressor']:
        #     self.extension = "_clf.sav"
        # elif self.model_config['ensemble_config']['classifier'] and self.model_config['ensemble_config']['regressor']:
        #     self.extension = ".sav"
        # else:
        #     self.extension = "_reg.sav"

        if self.model_config['ensemble_config']['classifier'] and not self.model_config['ensemble_config']['regressor']:
            self.model_type = "clf"
        elif self.model_config['ensemble_config']['classifier'] and self.model_config['ensemble_config']['regressor']:
            self.model_type = "zir"
        if self.model_config['ensemble_config']['regressor'] and not self.model_config['ensemble_config']['classifier']:
            self.model_type = "reg"

        self.extension = "_" + self.model_type + ".sav"

       
    def merge_performance(self, model):
        
        all_performance = []

        for i in range(len(self.d.columns)):
            target = self.d.columns[i]
            target_no_space = target.replace(' ', '_')
            with open(self.root + self.model_config['path_out'] + self.model_config['run_name'] + "/scoring/" + model + "/" + target_no_space + self.extension, 'rb') as file:
                m = pickle.load(file)
            
            if self.model_config['ensemble_config']['classifier'] and not self.model_config['ensemble_config']['regressor']:
                #estimate performance of classifier
                performance = pd.DataFrame({'target':[target]})
                all_performance.append(performance)
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
            os.makedirs(self.root + self.model_config['path_out'] + self.model_config['run_name'] + "/posts/performance/")
        except:
            None
        all_performance.to_csv(self.root + self.model_config['path_out'] + self.model_config['run_name'] + "/posts/performance/" + model + "_performance.csv", index=False)

        # if configuration==None:
        #     all_performance.to_csv(self.root + self.model_config['path_out'] + model + "_performance.csv", index=False)
        # else:
        #     all_performance.to_csv(self.root + self.model_config['path_out'] + model + "_" + configuration + "_performance.csv", index=False)
        
        print("finished merging performance")

    def merge_parameters(self, model):
        
        all_parameters = []

        for i in range(len(self.d.columns)):
            
            target = self.d.columns[i]
            target_no_space = target.replace(' ', '_')

            with open(self.root + self.model_config['path_out'] + self.model_config['run_name'] + "/model/" + model + "/" + target_no_space + self.extension, 'rb') as file:
                m = pickle.load(file)

            if self.model_type == "reg":

                if model == "rf":
                    max_depth = m.regressor_.named_steps.estimator.max_depth
                    max_features = m.regressor_.named_steps.estimator.max_features
                    max_samples = m.regressor_.named_steps.estimator.max_samples
                    min_samples_leaf = m.regressor_.named_steps.estimator.min_samples_leaf
                    parameters = pd.DataFrame({'target':[target], 'max_depth':[max_depth], 'max_features':[max_features], 
                                            'max_samples':[max_samples], 'min_samples_leaf':[min_samples_leaf]})
                    all_parameters.append(parameters)
                elif model == "xgb":
                    max_depth = m.regressor_.named_steps.estimator.max_depth
                    subsample = m.regressor_.named_steps.estimator.subsample
                    colsample_bytree = m.regressor_.named_steps.estimator.colsample_bytree

                    learning_rate = m.regressor_.named_steps.estimator.learning_rate
                    alpha = m.regressor_.named_steps.estimator.reg_alpha

                    parameters = pd.DataFrame({'target':[target], 'max_depth':[max_depth], 'subsample':[subsample], 'colsample_bytree':[colsample_bytree],
                                            'learning_rate':[learning_rate], 'alpha':[alpha]                                           
                                            })
                    all_parameters.append(parameters)
                elif model == "knn":
                    max_features = m.regressor_.named_steps.estimator.max_features
                    max_samples = m.regressor_.named_steps.estimator.max_samples

                    leaf_size = m.regressor_.named_steps.estimator.estimator.leaf_size
                    n_neighbors = m.regressor_.named_steps.estimator.estimator.n_neighbors
                    p = m.regressor_.named_steps.estimator.estimator.p
                    weights = m.regressor_.named_steps.estimator.estimator.weights

                    parameters = pd.DataFrame({'target':[target], 'max_features':[max_features], 'max_samples':[max_samples],
                                            'leaf_size':[leaf_size], 'p':[p], 'n_neighbors':[n_neighbors], 'weights':[weights]
                                            })
                    all_parameters.append(parameters) 

            elif self.model_type == "clf":
                #still to implement!
                parameters = pd.DataFrame({'target':[target]})
                all_parameters.append(parameters) 

            elif self.model_type == "zir":
                #still to implement!
                parameters = pd.DataFrame({'target':[target]})
                all_parameters.append(parameters) 

        all_parameters= pd.concat(all_parameters)
        try: #make new dir if needed
            os.makedirs(self.root + self.model_config['path_out'] + self.model_config['run_name'] + "/posts/parameters/")
        except:
            None
        all_parameters.to_csv(self.root + self.model_config['path_out'] + self.model_config['run_name'] + "/posts/parameters/" + model + "_parameters.csv", index=False)
        
        print("finished merging parameters")



    def estimate_carbon(self, variable):

        """
        Estimate carbon content for each species


        Parameters
        ----------

        variable: string
            carbon content to estimate

        """

        w = self.traits.query('Target in @self.targets')
        var = w[variable].to_numpy()
        print(var)
        self.d = self.d.apply(lambda row : (row[self.targets]* var), axis = 1)
        print("finished estimating " + variable)

    def def_groups(self, dict):
        """
        Define groups of species

        Parameters
        ----------

        dict: dictionary
        A dictionary containing group definitions

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

        variable: string
            variable that is used to estimate cwm.

        """

        w = self.traits.query('Target in @self.targets')
        var = w[variable].to_numpy()
        var_name = 'cwm ' + variable
        self.d[var_name] = self.d.apply(lambda row : np.average(var, weights=row[self.targets]), axis = 1)
        print("finished calculating CWM " + variable)

    def diversity(self):
        self.d['shannon'] = self.d.apply(shannon, axis=1)
        print("finished calculating shannon diversity")

    def total(self):
        """
        Sum target rows to estimate total.

        Notes
        ----------
        Useful for estimating total species abundances if targets are continuous.
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
                Prefix conversion, e.g., umol to Tmol would be 1e-21, default is 1 (no conversion).
            
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
            
            ds = self.parent.d.to_xarray()
            vol_conversion = self.vol_conversion
            magnitude_conversion = self.magnitude_conversion
            molar_mass = self.molar_mass
            rate = self.rate

            days_per_month = 365.25 / 12  # approx days/month


            if subset_depth:
                ds = ds.sel(depth=slice(0, subset_depth))
            
            if rate:
                total = (ds[variable] * ds['volume'] * days_per_month).sum(dim=['lat', 'lon', 'depth', 'time'])
                total = (total / molar_mass) * vol_conversion * magnitude_conversion
            else:
                total = (ds[variable] * ds['volume']).sum(dim=['lat', 'lon', 'depth', 'time'])
                total = (total / molar_mass) * vol_conversion * magnitude_conversion

            if monthly:
                total /= 12
            
            print("Final integrated total:", total.values)
            return total


        def integrated_totals(self, targets, monthly=False, subset_depth=None, 
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
            if "total" in ds:
                targets = np.append(targets, 'total')
            totals = []
    
            for target in targets:
                try:
                    print(target)
                    total = self.integrate_total(variable=target, monthly=monthly, subset_depth=subset_depth)
                    total_df = pd.DataFrame({'total': [total.values], 'variable': target})
                    totals.append(total_df)
                except Exception as e:
                    print(f"Some targets do not have predictions! Missing: {target}")
                    print(f"Error: {e}")

            totals = pd.concat(totals)

            if export:
                depth_str = f"_depth_{subset_depth}m" if subset_depth else ""
                avg_str = "_monthly_avg" if monthly else ""
                try: #make new dir if needed
                    os.makedirs(self.parent.root + self.parent.model_config['path_out'] + self.parent.model_config['run_name'] + "/posts/integrated_totals/")
                except:
                    None
                totals.to_csv(self.parent.root + self.parent.model_config['path_out'] + self.parent.model_config['run_name'] + "/posts/integrated_totals/" + model + '_integrated_totals_PI' + self.parent.pi + depth_str + avg_str + ".csv", index=False)
                print(f"Exported totals")     


    def merge_env(self, X_predict):
        """
        Merge model output with environmental data 
        """

        def concat(d, X_predict, chunk_size=1000):
            # Ensure the length of X_predict is divisible by chunk_size
            for start in range(0, len(X_predict), chunk_size):
                end = start + chunk_size
                chunk = X_predict[start:end]
                
                d = pd.concat([d, chunk], axis=1)
                
                # Delete chunk and run garbage collection to free up memory
                del chunk
                gc.collect()
                
            return d

        self.d = concat(self.d, X_predict)

    def return_d(self):
        return(self.d)

    def export_ds(self, file_name, 
                  author=None, description=None):
        """
        Export processed dataset to netcdf.

        Parameters
        ----------
        file_name: name netcdf will be saved as. 
        author: author included in netcdf description
        description: title included in netcdf description

        Notes
        ----------
        data export location is defined in the model_config.yml

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
        ds.to_netcdf(self.path_out + file_name + "_PI" + self.pi + ".nc")
        print("exported ds to: " + self.path_out + file_name + "_PI" + self.pi + ".nc")
        #add nice metadata


    def export_csv(self, file_name):
        """
        Export processed dataset to csv.

        Parameters
        ----------
        file_name: name csv will be saved as. 

        Notes
        ----------
        data export location is defined in the model_config.yml

        """
    
        try: #make new dir if needed
            os.makedirs(self.path_out)
        except:
            None
    
        print(self.d.head())
        self.d.to_csv(self.path_out + file_name + "_PI" + self.pi + ".csv")
        print("exported d to: " + self.path_out + file_name + "_PI" + self.pi + ".csv")
        #add nice metadata
