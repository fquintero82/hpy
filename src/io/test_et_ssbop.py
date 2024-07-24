import numpy as np
import pandas as pd

from os.path import isfile

import xarray as xr
import time as time
import rioxarray

f = 'E:/projects/hpy/examples/hydrosheds/conus_centroids.csv'
df = pd.read_csv(f)
yy = xr.DataArray(df['y'].to_numpy(dtype=np.float32))
xx = xr.DataArray(df['x'].to_numpy(dtype=np.float32))

f = 'E:/projects/rush/et/2008/det2008001.modisSSEBopETactual/det2008001.modisSSEBopETactual.tif'

nrow = len(df)+1
nt = 366
output = np.zeros( dtype=np.float32, shape=(nrow,nt))
fileout= 'E:/projects/rush/et/2008/et.npy'
t=0
for i in range(1,366):
    print(i)
    dd =str(i).zfill(3)
    f = 'E:/projects/rush/et/2008/det2008%s.modisSSEBopETactual.tif'%dd
    if isfile(f) == False:
        break
    ds = rioxarray.open_rasterio(f)
    subset = ds.sel(y=yy,x=xx,method='nearest')
    #var ='ET'
    #subset = subset[var]
    subset = subset.load()    
    aux = subset.to_numpy()
    ds.close()
    aux[aux==9999]=0
    aux = aux/100.
    print(f'Variable size: {aux.nbytes/1e6:.1f} MB')
    aux = np.nan_to_num(aux,copy=False,posinf=0.0,neginf=0.0)
    output[1:,t]=aux
    t+=1

np.save(file=fileout,arr=output)