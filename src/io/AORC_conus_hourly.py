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
    lid = None
    val = 0
    root,varname = access_file(time,options)
    if root is None:
        return lid , val
    if root is not None:
        lut = get_lid_xy(options)
        lid, val = extract_vals_from_var(root,varname,time,lut)
    return lid,val


def access_file(time:int, options=None):
    root = None
    varname = None
    if options is not None:
        dir = options['path']
    if 'prefix' in list(options.keys()):
        prefix = options['prefix']
    if 'varname' in list(options.keys()):
        varname = options['varname']
    
    d1 = datetime.fromtimestamp(time,pytz.UTC)
    month = '{:02d}'.format(d1.month)
    filein = os.path.join(dir,prefix+str(month)+'.nc')
    try:    
        root= Dataset(filein, mode='r')
    except FileNotFoundError as e:
        print(e)
    if varname not in root.variables:
        print('Error. variable {varname} not found in {filein}'.format(varname,filein))
        quit()
    return root, varname

def get_lid_xy(options=None):
    fcentroids = None
    format1 = 'csv'
    if options is not None:
        if 'centroids' in list(options.keys()):
            fcentroids = options['centroids']
            if isfile(fcentroids)==False:
                print('Error. Not valid path to hillslope centroids in yaml file.')
                quit()
        if 'format' in list(options.keys()):
            format1 = options['format']
    if fcentroids is None:
        print('Error. Hillslope centroids missing in yaml file.')
        quit()
    if format1 =='csv':
        df = pd.read_csv(fcentroids)
    if format1 == 'parquet':
        df = pd.read_parquet(fcentroids)
    df.columns = ['lid','x','y']
    return df


def extract_vals_from_var(nc:Dataset,
                          varname:str,
                          t:int,
                          df:pd.DataFrame):
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    lid = df['lid'].to_numpy()
    # Get the nc row, col for the point's lat, lon
    col = np.argmin(np.abs(nc.variables["lon"][:] - x))
    row = np.argmin(np.abs(nc.variables["lat"][:] - y))
    idx_time = np.argmin(np.abs(nc.variables["time"][:] - t))
    # Return a np.array with the netcdf data
    val = np.ma.getdata(nc.variables[varname][idx_time, row, col])
    return lid,val
