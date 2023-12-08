import pandas as pd
import numpy as np
from  os.path import isfile
from netCDF4 import Dataset
from models.model400names import CF_UNITS , STATES_NAMES, PARAM_NAMES
import time as mytime

def get_time(timedimension,timevariable,time_to_ingest):
    #gets the position in the timedimension to add the data
    # if the time already exists in the time variable, it will overwrite the values
    current_len = len(timedimension)
    flag = np.isin([time_to_ingest],timevariable)[0]
    if flag:
        current_len = np.argwhere(time_to_ingest==timevariable)[0][0]
    return current_len

def save_to_netcdf(states:pd.DataFrame,params:pd.DataFrame,time:int,filename:str,discharge_only=False):
    t = mytime.time()
    if isfile(filename) == False:
        create_empty_ncdf(states,params,filename)
    try:    
        with Dataset(filename, mode='r+') as root:
            # Access the unlimited dimension
            # unlimited_dim = root.dimensions['timedim']
            # Get the current length of the unlimited dimension
            # current_len = len(unlimited_dim)
            current_len = get_time(root.dimensions['timedim'],root['time'][:],time)
            # add new time to file
            root['time'][current_len] = time
            if discharge_only==False:
                # Add the new data to the existing variable
                n = np.array(states.columns,dtype=np.str_)
                for ii in range(1,len(states.columns)):
                    root['states/'+n[ii]][current_len,:] = states[n[ii]]
            if discharge_only==True:
                root['states/discharge'][current_len,:] = states['discharge']
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
    x = int(1000*(mytime.time()-t))
    print('saved to netcdf in {x} msec'.format(x=x))

def create_empty_ncdf(states:pd.DataFrame,params:pd.DataFrame,filename:str,discharge_only=False):
    try:
        fn = filename
        root = Dataset(fn, 'w', format='NETCDF4')
        nlinks = states.shape[0]
        nstates = states.shape[1]
        root.createDimension('linkdim', nlinks)
        root.createDimension('timedim', None)
        n = np.array(states.columns,dtype=np.str_)
        nparams = np.array(params.columns,dtype=np.str_)
        
        # create variables
        # params variables
        for ii in range(1,len(params.columns)):
            var = root.createVariable(varname='params/'+nparams[ii],
                                        datatype = PARAM_NAMES[nparams[ii]],
                                        dimensions=('linkdim'), 
                                        chunksizes=None,
                                        fill_value=float('nan'), 
                                        zlib=True
                                        )
            var.units = CF_UNITS['params.'+nparams[ii]]
            var.setncatts({'units':CF_UNITS['params.'+nparams[ii]]})
            root['params/'+nparams[ii]][:] = np.array(params[nparams[ii]],dtype=np.float32)
        
        # state variables
        if discharge_only==False:
            for ii in range(1,len(states.columns)):
                var_state = root.createVariable(varname='states/'+n[ii],
                                            datatype =STATES_NAMES[n[ii]],
                                            dimensions = ('timedim','linkdim'), #unlimited dimension is leftmost (recommended)
                                            chunksizes=(1,nlinks),
                                            fill_value=float('nan'), 
                                            zlib=True
                                            )
                var_state.units = CF_UNITS['states.'+n[ii]]
                var_state.setncatts({'units':CF_UNITS['states.'+n[ii]]})

        if discharge_only==True:
            var_state = root.createVariable(varname='states/discharge',
                                            datatype =STATES_NAMES['discharge'],
                                            dimensions = ('timedim','linkdim'), #unlimited dimension is leftmost (recommended)
                                            chunksizes=(1,nlinks),
                                            fill_value=float('nan'), 
                                            zlib=True
                                            )
            var_state.units = CF_UNITS['states.discharge']
            var_state.setncatts({'units':CF_UNITS['states.discharge']})
            #linkid variable
        root.createVariable('link_id',np.uint32,('linkdim'),fill_value=-1,zlib=True)
        root['link_id'][:] = np.array(states['link_id'],dtype=STATES_NAMES['link_id'])

        #time variable
        root.createVariable('time',np.uint32,('timedim'),fill_value=0,zlib=True)
        # close the netcdf file
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
