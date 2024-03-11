from hlm import HLM
import pandas as pd
from utils.network.network_subset import get_basin_ids
from utils.params.params_default import get_default_params
import numpy as np
import sys

#code to create prm files for model 404
def get_base_df(config_file):
    instance = HLM()
    #config_file = 'examples/hydrosheds/conus_example.yaml'
    #config_file = 'examples/small/small.yaml'
    instance.init_from_file(config_file)
    df1 = instance.network.loc[:,['area_hillslope','channel_length','drainage_area']]
    df1['area_hillslope']=df1['area_hillslope']/1e6
    df1['channel_length']=df1['channel_length']/1e3
    df1['drainage_area']=df1['drainage_area']/1e6
    
    #here are added the model parameters columns
    df = pd.concat([df1,instance.params.iloc[:,1:]],axis=1)
    return df

def save_df(df,fileout):
    #fileout = '/Users/felipe/tmp/conus_distrib_param.prm'
    #fileout = '/Users/felipe/tmp/small.prm'
    a = df.to_csv(fileout,sep=' ',header=True,index=True)

def get_link_subbasin(network,outlet):
    idx_upstream_links = network['idx_upstream_link'].to_numpy()
    wh = np.where(network['link_id'].to_numpy()==outlet)[0][0]
    idxs = network['idx'].to_numpy()
    id = idxs[wh]
    sys.setrecursionlimit(int(1E6))
    x = get_basin_ids(id,idx_upstream_links)
    return x

def test_iowa_model400():
    df = pd.read_pickle('/Users/felipe/tmp/iowa_operational/ifis_iowa.pkl')
    df1 = df.loc[:,['area_hillslope','channel_length','drainage_area']]
    df1['area_hillslope']=df1['area_hillslope']/1e6
    df1['channel_length']=df1['channel_length']/1e3
    df1['drainage_area']=df1['drainage_area']/1e6
    params = get_default_params(df)
    df3 = pd.concat([df1,
                     params.iloc[:,1:]
                     ]
                     ,axis=1)
    calibration = pd.read_csv('/Users/felipe/tmp/iowa_operational/calibration_from_maps.csv')
    calibration.index = calibration['HYBAS_ID']

    lookup = pd.read_csv('/Users/felipe/tmp/iowa_operational/iowa_network_hydrosheds_level5.csv')
    #lookup.index = lookup['LINKNO']
    n = len(lookup)
    for i in range(n):
        print(i)
        mylink = lookup.iloc[i,0]
        mybasin = lookup.iloc[i,1]
        myparams = calibration.loc[mybasin]
        df3.loc[mylink,'river_velocity'] =myparams['vo']
        df3.loc[mylink,'lambda1'] =myparams['l1']
        df3.loc[mylink,'lambda2'] =myparams['l2']
        df3.loc[mylink,'max_storage'] =myparams['Hu']
        df3.loc[mylink,'infiltration'] =myparams['Infil']
        df3.loc[mylink,'percolation'] =myparams['perc']
        df3.loc[mylink,'surface_velocity'] =myparams['surf_vl']
        df3.loc[mylink,'tr_subsurface'] =myparams['rs_sbsr']
        df3.loc[mylink,'tr_groundwater'] =myparams['res_gw']
        df3.loc[mylink,'melt_factor'] =myparams['mlt_fct']
        fileout = '/Users/felipe/tmp/iowa_operational/iowa_operational_model404.prm'
        a = df3.to_csv(fileout,sep=' ',header=True,index=True)
def run():
    config_file ='/Users/felipe/tmp/iowa_operational/iowa_operational_imac_dischargeonly.yaml'
    df = get_base_df(config_file)
    print('lolo')

    fileout = '/Users/felipe/tmp/iowa_operational/iowa_operational_model404.prm'
    save_df(df,fileout)

run()