import pandas as pd
import numpy as np
from model400names import STATES_NAMES
import sys
from  os.path import isfile
from netCDF4 import Dataset
from utils.states.states_default import get_default_states



def states_from_nc(ncfile:str,unixtime:np.int32,network:pd.DataFrame):
    check, idx_unixtime = check_file(ncfile,unixtime)
    df = get_default_states(network)
    root = Dataset(ncfile, mode='r')
    n = np.array(df.columns,dtype=np.str_)
    for ii in range(1,len(df.columns)):
        df[n[ii]]= root['states/'+n[ii]][idx_unixtime,:]
    return df

def check_file(ncfile,unixtime:np.int32):
    check=True
    check = isfile(ncfile)
    if check == False:
        print('file %s doesnt exists. check yaml file'%ncfile)
        quit()
    root = Dataset(ncfile, mode='r')
    time = root['time'][:]
    check = unixtime in time
    root.close()
    if check == False:
        print('unixtime %s not in nc file. '%unixtime)
        quit()
    wh = np.where(time == unixtime)[0][0]
    return check, wh
