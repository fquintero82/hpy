import os
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from os.path import isfile
from netCDF4 import Dataset
#import geopandas as gpd
import xarray as xr
import time as time
from utils.forcings.forcing_manager import forcing

#dir = '/Users/felipe/tmp/aorc/'
#time = 1199145600
#prefix = 'AORC_APCP_2008'
prefix = ''

class AORC_conus_hourly(forcing):
    pass

def get_values(unixtime: int,options=None):
    lid = None
    val = 0
    #root,varname = access_file(time,options)
    ncfile = get_ncfile(unixtime,options)
    # if root is None:
    #     print('Error. Could not read netcdf file in yaml file')
    #     quit()
    if ncfile is not None:
        lut = get_lid_xy(options)
        #nc2geopandas(root,varname,time)
        d1 = datetime.fromtimestamp(unixtime,pytz.UTC).isoformat()
        d1 = d1.split('+')[0]
        lid, val = nc2xr(ncfile,lut,d1)
    return lid,val

def get_relative_time(unixtime:int):
    d1 = datetime.fromtimestamp(unixtime,pytz.UTC)
    begin = datetime(d1.year,d1.month,1,0).replace(tzinfo=pytz.timezone("UTC")).timestamp()
    hours_elapsed = round((unixtime - begin)/3600)
    return hours_elapsed


def get_ncfile(time:int, options=None):
    #finds what aorc netcdf file should be open based on the unixtime of the time simulation
    root = None
    varname = None
    if options is not None:
        dir = options['path']
    if 'prefix' in list(options.keys()):
        prefix = options['prefix']
    d1 = datetime.fromtimestamp(time,pytz.UTC)
    month = '{:02d}'.format(d1.month)
    filein = os.path.join(dir,prefix+str(month)+'.nc')
    return filein

# def access_file(time:int, options=None):
#     #reads the aorc netcdf file that should be opened based on the unixtime of the time simulation
#     root = None
#     varname = None
#     if options is not None:
#         dir = options['path']
#     if 'prefix' in list(options.keys()):
#         prefix = options['prefix']
#     if 'varname' in list(options.keys()):
#         varname = options['varname']
#     if varname is None:
#         print('Error. precipitation varname not in yaml file')
#         quit()
#     d1 = datetime.fromtimestamp(time,pytz.UTC)
#     month = '{:02d}'.format(d1.month)
#     filein = os.path.join(dir,prefix+str(month)+'.nc')
#     try:    
#         root= Dataset(filein, mode='r')
#     except FileNotFoundError as e:
#         print(e)
#     if varname not in root.variables:
#         print('Error. variable {varname} not found in {filein}'.format(varname,filein))
#         quit()
#     return root, varname

def get_lid_xy(options=None):
    fcentroids = None
    format1 = 'csv'
    if options is not None:
        if 'centroids' in list(options.keys()):
            fcentroids = options['centroids']
            if isfile(fcentroids)==False:
                print('Error. Not valid path to hillslope centroids in yaml file.')
                quit()
    if fcentroids is None:
        print('Error. Hillslope centroids missing in yaml file.')
        quit()
    _, extension = os.path.splitext(fcentroids)
    if extension =='.pkl':
        df = pd.read_pickle(fcentroids)
    if extension =='csv':
        df = pd.read_csv(fcentroids)    
    df.columns = ['lid','x','y']
    return df

# def get_col_row(nc:Dataset,df:pd.DataFrame):
#     x = df['x'].to_numpy()
#     y = df['y'].to_numpy()
#     lid = df['lid'].to_numpy()
#     n = len(lid)
#     col = np.zeros(shape=(n),dtype=np.int32)
#     row = np.zeros(shape=(n),dtype=np.int32)
#     # Get the nc row, col for the point's lat, lon
#     xnc = np.array(nc.variables["lon"][:])
#     ync = np.array(nc.variables["lat"][:])
#     #need vectorized version of this function. 

#     for ii in np.arange(n):
#         print(ii)
#         col[ii] = np.argmin(np.abs(xnc - x[ii]))
#         row[ii] = np.argmin(np.abs(ync - y[ii]))
#     df['col']=  col
#     df['row'] = row
#     return df


# def nc2geopandas(nc:Dataset,varname:str,t:int):
#     idx_time = get_relative_time(t)
#     #idx_time = np.argwhere(nc.variables["time"][:]==reltime)
#     data = nc[varname][idx_time,:,:]
#     gdf = gpd.GeoDataFrame(data, crs=nc.crs)
#     #values = gdf.loc[gdf.geometry.contains(coordinates)][column_name]

def nc2xr(ncfile:str,df:pd.DataFrame,t:str):
    nc = xr.open_dataset(ncfile)
    lid = df['lid'].to_numpy()
    crd_ix = df.set_index('lid').to_xarray()
    out = nc.sel(lon=crd_ix.x,lat=crd_ix.y,time=t,method='nearest')
    val = out['APCP'].to_numpy()
    return lid,val
    

# def extract_vals_from_var(nc:Dataset,
#                           varname:str,
#                           t:int,
#                           df:pd.DataFrame):
#     x = df['x'].to_numpy()
#     y = df['y'].to_numpy()
#     lid = df['lid'].to_numpy()
#     idx_time = time2hours(t)
#     #idx_time = np.argmin(np.abs(nc.variables["time"][:] - t))
#     # Return a np.array with the netcdf data
#     pass
#     #return lid,val

def test2():
    ncfile = '/Users/felipe/tmp/aorc/AORC_APCP_200801.nc'
    t = time.time()
    nc = xr.open_dataset(ncfile)
    val = nc.sel(lon=125.1,lat=52.5,time='2008-01-01T01:00:00',method='nearest').to_array()
    print(time.time()-t)

def test3():
    fcentroids = 'examples/iowa/centroids.csv'
    t = time.time()
    df = pd.read_csv(fcentroids)
    print(time.time()-t)
    df.to_pickle('examples/iowa/centroids.pkl')
    fcentroids = 'examples/iowa/centroids.pkl'
    t = time.time()
    df = pd.read_pickle(fcentroids)
    print(time.time()-t)

   

def test():
    import yaml
    from yaml import Loader
    config_file = 'examples/hydrosheds/conus_example.yaml'
    stream =open(config_file)
    d = yaml.load(stream,Loader=Loader)
    options = d['forcings']['precipitation']
    time = 1199145600
    lid,val = get_values(time,options)
# https://github.com/pydata/xarray/issues/1385


    