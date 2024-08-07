# import required packages
import sys
from yaml import load
from yaml import CLoader as Loader
from abil.post import post
import pandas as pd

from datetime import datetime
current_date = datetime.today().strftime('%Y-%m-%d')


try:
    print(sys.argv[0])
    with open('/user/work/mv23682/Abil/wiseman2024/ensemble_regressor.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = True
    root = model_config['hpc_root']

except:
    with open('/home/mv23682/Documents/Abil/wiseman2024/ensemble_regressor.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = False
    root = model_config['local_root']


X_predict =  pd.read_csv(root + model_config['env_data_path'])
X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)

m = post(model_config)
m.merge_performance(model="ens")
m.merge_performance(model="xgb", configuration= "reg")
m.merge_performance(model="rf", configuration= "reg")
m.merge_performance(model="knn", configuration= "reg")
#m.merge_performance(model="mlp", configuration="reg")

m.merge_parameters(model="rf")
m.merge_parameters(model="xgb")
m.merge_parameters(model="knn")
#m.merge_parameters(model="mlp")

#m.merge_env(X_predict) #comment out to save space
m.export_ds(current_date + "_cp_ci50")
m.export_csv(current_date + "_cp_ci50")

m = post(model_config, ci=5)
m.export_ds(current_date + "_cp_ci5")
m.export_csv(current_date + "_cp_ci5")

m = post(model_config, ci=95)
m.export_ds(current_date + "_cp_ci95")
m.export_csv(current_date + "_cp_ci95")

