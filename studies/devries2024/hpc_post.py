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
    with open('/user/work/ba18321/Abil/devries2024/2-phase.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = True
    root = model_config['hpc_root']

except:
    with open('/home/phyto/Abil/devries2024/2-phase.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = False
    root = model_config['local_root']

print("path:")
print(root + model_config['targets'])
targets = pd.read_csv(root + model_config['targets'])
#print("targets: ")
#print(targets)
targets = targets[targets['n']>85]
targets =  targets['Target'].values
depth_w = 5
conversion = 1e3 #L-1 to m-3

# m = post(model_config)
# m.merge_performance(model="ens")
# m.merge_performance(model="xgb", configuration= "reg")
# m.merge_performance(model="rf", configuration= "reg")
# m.merge_performance(model="knn", configuration= "reg")
# m.merge_parameters(model="rf")
# m.merge_parameters(model="xgb")
# m.merge_parameters(model="knn")
# m.total()
# m.integrated_totals(targets, depth_w =depth_w,
#                     conversion=conversion,
#                     model="abundance_ci50")

# #m.merge_env(X_predict)
# m.export_ds(current_date + "_abundance_ci50")
# m.export_csv(current_date + "_abundance_ci50")

def do_post(ci=50, datatype="pg poc", diversity=False):

    if datatype=="pg poc":
        dataset="POC_ci" + str(ci)
    elif datatype=="pg pic":
        dataset="PIC_ci" + str(ci)
    else:
        raise ValueError("undefined datatype")

    print("starting " + dataset)
    m = post(model_config, ci=ci)
    m.estimate_carbon(datatype)
    print("finished estimating" + datatype)
    if diversity:
        m.diversity()
    m.total()
    print("finished estimating total")
    m.export_ds(current_date + "_" + dataset)
    m.export_csv(current_date + "_" + dataset)
    m.integrated_totals(targets, depth_w =depth_w,
                        conversion=conversion)

    m.integrated_totals(targets, depth_w =depth_w,
                        conversion=conversion, subset_depth=100)
    
    m = None
    print("finished post for: " + dataset)


do_post(ci=50, datatype="pg poc", diversity=True)
do_post(ci=50, datatype="pg pic")

do_post(ci=5, datatype="pg poc", diversity=True)
do_post(ci=5, datatype="pg pic")

do_post(ci=95, datatype="pg poc", diversity=True)
do_post(ci=95, datatype="pg pic")
