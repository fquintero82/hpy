import pandas as pd
import numpy as np
from  os.path import isfile
from netCDF4 import Dataset
from model400names import CF_UNITS , STATES_NAMES


def save_to_netcdf(states:pd.DataFrame,time:int,filename:str):
    if isfile(filename) == False:
        create_empty_ncdf(states,filename)
    try:    
        with Dataset(filename, mode='r+') as data:
            # Access the unlimited dimension
            unlimited_dim = data.dimensions['timedim']
            #nlinks = len(data.dimensions['linkdim'])
            #nstates = len(data.dimensions['statedim'])
            # Get the current length of the unlimited dimension
            current_len = len(unlimited_dim)
            # Add the new data to the existing variable
            #data.variables['state'][0:nlinks-1,0:nstates-1,0] = np.float32(1)
            n = np.array(states.columns,dtype=np.str_)
            for ii in range(1,len(states.columns)):
                data.variables[n[ii]][current_len,:] = states[n[ii]]
    except OSError as e:
        print('Error . NETCDF file is open by another program.')
        quit()

def create_empty_ncdf(states:pd.DataFrame,filename:str):
    fn = filename
    root = Dataset(fn, 'w', format='NETCDF4')
    nlinks = states.shape[0]
    nstates = states.shape[1]
    root.createDimension('linkdim', nlinks)
    #root.createDimension('statedim', nstates)
    root.createDimension('timedim', None)
    #var_state_name = root.createVariable('state_name', np.str_, ('statedim',))
    n = np.array(states.columns,dtype=np.str_)
    #root['state_name'][:] = n
    
    for ii in range(1,len(states.columns)):
        var_state = root.createVariable(n[ii],
                                    STATES_NAMES[n[ii]],
                                    ('timedim','linkdim'), #unlimited dimension is leftmost (recommended)
                                    chunksizes=(1,nlinks),
                                    fill_value=float('nan'), 
                                    zlib=True
                                    )
        var_state.units = CF_UNITS['states.'+n[ii]]
    #only for linkid
    varlid = root.createVariable('link_id',np.uint32,('linkdim'),fill_value=-1,zlib=True)
    root['link_id'][:] = np.array(states['link_id'],dtype=STATES_NAMES['link_id'])
    root.close()

def save_to_pickle(states:pd.DataFrame,time:int):
    f = 'examples/cedarrapids1/out/{}.pkl'.format(time)
    states.to_pickle(f)

def read_from_pickle(states:pd.DataFrame,fileinput:str):
    pass
