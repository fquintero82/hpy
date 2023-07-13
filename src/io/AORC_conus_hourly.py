import os
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from os.path import isfile
from netCDF4 import Dataset

dir = '/Users/felipe/tmp/aorc/'
time = 1199145600
prefix = 'AORC_APCP_2008'
prefix = ''
def get_values(time: int,options=None):
    if options is not None:
        dir = options['path']
    if 'prefix' in list(options.keys()):
        prefix = options['prefix']
    d1 = datetime.fromtimestamp(time,pytz.UTC)
    month = '{:02d}'.format(d1.month)
    filein = os.path.join(dir,prefix+str(month)+'.nc')
    lid = None
    val = 0
    try:    
        with Dataset(filein, mode='r') as root:
            pass
    except FileNotFoundError as e:
        print(e)
    return lid , val     

def get_lid_xy(options=None):
    fcentroids = None
    if options is not None:
        if 'centroids' in list(options.keys()):
            fcentroids = options['centroids']
            if isfile(fcentroids)==False:
                print('Error. Not valid path to hillslope centroids in yaml file.')
                quit()
    if fcentroids is None:
        print('Error. Hillslope centroids missing in yaml file.')
        quit()
    df = pd.read_csv(fcentroids)
def ExtractVarsFromNetcdf(point, ncdir, varnames):
    """   
    @params:
        point      - Required : shapely point
        ncdir      - Required : The directory of the netcdf file.
        varnames   - Required : The netcdf variables
    """

    with Dataset(ncdir, "r") as nc:

        # Get the nc row, col for the point's lat, lon
        col = np.argmin(np.abs(nc.variables["lon"][:] - point.x))
        row = np.argmin(np.abs(nc.variables["lat"][:] - point.y))

        # Return a np.array with the netcdf data
        nc_data = np.ma.getdata(
            [nc.variables[varname][:, row, col] for varname in varnames]
        )

        return nc_data

def read_bin(path):
    return np.fromfile(path,
                    dtype=np.dtype([('lid', np.int32),('val', np.float32)]),
                    offset=4,) 



