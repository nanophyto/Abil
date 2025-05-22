# import required packages
import sys
from yaml import load
from yaml import CLoader as Loader
from abil.post import post
import pandas as pd


try:
    print(sys.argv[0])
    with open('/user/work/ba18321/Abil/configuration/2-phase_ensemble_cluster.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = True
    root = model_config['hpc_root']

except:
    with open('/home/phyto/Abil/configuration/2-phase_ensemble_cluster.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = False
    root = model_config['local_root']

print("path:")
print(root + model_config['targets'])
targets = pd.read_csv(root + model_config['targets'])
targets =  targets['Target'].values
depth_w = 5
vol_conversion = 1e3 #L-1 to m-3

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

    integ = m.integration(m,vol_conversion=vol_conversion)
    integ.integrated_totals(targets)
    integ.integrated_totals(targets, subset_depth=100)
    
    m = None
    print("finished post for: " + dataset)


do_post(ci=50, datatype="pg poc", diversity=True)
do_post(ci=50, datatype="pg pic")

do_post(ci=5, datatype="pg poc", diversity=True)
do_post(ci=5, datatype="pg pic")

do_post(ci=95, datatype="pg poc", diversity=True)
do_post(ci=95, datatype="pg pic")
