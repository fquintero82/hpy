import pandas as pd
import numpy as np
from  os.path import isfile
from netCDF4 import Dataset
from model400names import CF_UNITS , STATES_NAMES, PARAM_NAMES


def save_to_netcdf(states:pd.DataFrame,params:pd.DataFrame,time:int,filename:str):
    if isfile(filename) == False:
        create_empty_ncdf(states,params,filename)
    try:    
        with Dataset(filename, mode='r+') as root:
            # Access the unlimited dimension
            unlimited_dim = root.dimensions['timedim']
            #nlinks = len(data.dimensions['linkdim'])
            #nstates = len(data.dimensions['statedim'])
            # Get the current length of the unlimited dimension
            current_len = len(unlimited_dim)
            # add new time to file
            #root.variables['time'][current_len] = time
            root['time'][current_len] = time
            # Add the new data to the existing variable
            #data.variables['state'][0:nlinks-1,0:nstates-1,0] = np.float32(1)
            n = np.array(states.columns,dtype=np.str_)
            for ii in range(1,len(states.columns)):
                #root.variables['states/'+n[ii]][current_len,:] = states[n[ii]]
                root['states/'+n[ii]][current_len,:] = states[n[ii]]
    except OSError as e:
        print(e)
        print('Error NETCDF file is open by another program.')
        quit()
    except KeyError as e:
        print(e)
        print('NETCDF file is corrupted')
        print('Delete the file and restart')
        quit()
    except IndexError as e:
        print(e)
        print('NETCDF file is corrupted')
        print('Delete the file and restart')
        quit()

def create_empty_ncdf(states:pd.DataFrame,params:pd.DataFrame,filename:str):
    try:
        fn = filename
        root = Dataset(fn, 'w', format='NETCDF4')
        nlinks = states.shape[0]
        nstates = states.shape[1]
        root.createDimension('linkdim', nlinks)
        root.createDimension('timedim', None)
        #var_state_name = root.createVariable('state_name', np.str_, ('statedim',))
        n = np.array(states.columns,dtype=np.str_)
        nparams = np.array(params.columns,dtype=np.str_)
        #root['state_name'][:] = n
        

        for ii in range(1,len(params.columns)):
            var_state = root.createVariable(varname='params/'+nparams[ii],
                                        datatype = PARAM_NAMES[nparams[ii]],
                                        dimensions=('linkdim'), 
                                        chunksizes=None,
                                        fill_value=float('nan'), 
                                        zlib=True
                                        )
            var_state.units = CF_UNITS['params.'+nparams[ii]]
            root['params/'+nparams[ii]][:] = np.array(params[nparams[ii]],dtype=np.float32)

        for ii in range(1,len(states.columns)):
            var_state = root.createVariable(varname='states/'+n[ii],
                                        datatype =STATES_NAMES[n[ii]],
                                        dimensions = ('timedim','linkdim'), #unlimited dimension is leftmost (recommended)
                                        chunksizes=(1,nlinks),
                                        fill_value=float('nan'), 
                                        zlib=True
                                        )
            var_state.units = CF_UNITS['states.'+n[ii]]
        #linkid variable
        root.createVariable('link_id',np.uint32,('linkdim'),fill_value=-1,zlib=True)
        root['link_id'][:] = np.array(states['link_id'],dtype=STATES_NAMES['link_id'])

        #time variable
        root.createVariable('time',np.uint32,('timedim'),fill_value=0,zlib=True)
        
        root.close()
    except KeyError as e:
        print(e)
        print('check that CF_UNITS for all variables are defined')
        quit()

def save_to_pickle(states:pd.DataFrame,time:int):
    f = 'examples/cedarrapids1/out/{}.pkl'.format(time)
    states.to_pickle(f)

def read_from_pickle(states:pd.DataFrame,fileinput:str):
    pass
