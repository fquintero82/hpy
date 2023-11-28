import pandas as pd
import numpy as np
from models.model400names import PARAM_NAMES
import sys

def params_from_prm_file_old(prm_file):
    df = pd.read_table(prm_file,
        header=None,
        names=['link_id','drainage_area','channel_length','area_hillslope'],
        sep=' ',
        skiprows=1,
        dtype={
            'link_id':np.uint64,
            'drainage_area':np.float32,
            'channel_length':np.float32,
            'area_hillslope':np.float32
        },
    )
    df.index = df['link_id']
    df['channel_length'] *= 1e3 # from km to m
    df['area_hillslope'] *= 1e6 #from km2 to m2
    df['drainage_area'] *= 1e6 #from km2 to m2
    df.info()
    return df

def params_from_prm_file(prm_file):
    data = np.loadtxt(prm_file,skiprows=1)
    df = pd.DataFrame(data)
    d = {'link_id':np.uint32,
         'drainage_area':np.float32,
         'channel_length':np.float32,
         'area_hillslope':np.float32
         }
    df.columns = list(d.keys())
    df = df.astype(d)
    df.index = df['link_id']
    df['channel_length'] *= 1e3 # from km to m
    df['area_hillslope'] *= 1e6 #from km2 to m2
    df['drainage_area'] *= 1e6 #from km2 to m2
    df.info()
    return df

def params_from_prm_file_split(prm_file)->pd.DataFrame:
    f = open(prm_file,'r')
    data = f.readlines()
    f.close()
    n = int(len(data))
    data = data[2:]
    x = data[0:n:3] #lids
    x = [int(a) for a in x] #lids
    y = data[1:n:3] #params
    n = len(y)
    da = np.empty(n)
    cl = np.empty(n)
    ah = np.empty(n)

    for i in range(n):
        items = y[i].split()
        da[i] = float(items[0])
        cl[i] = float(items[1])
        ah[i] = float(items[2])

    df = pd.DataFrame({
        'link_id': x,
        'drainage_area':da,
        'channel_length':cl,
        'area_hillslope':ah
    })
    
    d = {'link_id': np.uint32,
         'drainage_area':np.float32,
         'channel_length':np.float32,
         'area_hillslope':np.float32
         }
    # df.columns = list(d.keys())
    df = df.astype(d)
    df.index = df['link_id']
    df['channel_length'] *= 1e3 # from km to m
    df['area_hillslope'] *= 1e6 #from km2 to m2
    df['drainage_area'] *= 1e6 #from km2 to m2
    df.info()
    return df

def test():
    inputfile ='/Users/felipe/tmp/iowa_operational/ifis_iowa.prm'
    df = params_from_prm_file_split(inputfile)


