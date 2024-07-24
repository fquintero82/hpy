import numpy as np
import pandas as pd
from netCDF4 import Dataset

from os.path import isfile
from multiprocessing import Pool

import xarray as xr
import time as time

def fun(i):
    f = 'E:/projects/hpy/examples/hydrosheds/conus_centroids.csv'
    df = pd.read_csv(f)
    df['lid'] = df['HYRIV_ID'].to_numpy(dtype=np.int32) - int(7E7)
    crd_ix = df.set_index('lid').to_xarray()
    ncfile = 'E:/projects/aorc/CONUS_APCP/2008/AORC_APCP_200806.nc'
    nc = xr.open_dataset(ncfile)
    T = nc['time'].to_numpy()
    out = nc.sel(lon=crd_ix.x,lat=crd_ix.y,time= T[i], method='nearest')
    aux = out['APCP'].to_numpy()
    aux = np.nan_to_num(aux,copy=False,posinf=0.0,neginf=0.0)
    return(aux)

def run_map():
    timer = time.time()
    ncfile = 'E:/projects/aorc/CONUS_APCP/2008/AORC_APCP_200806.nc'
    nc = xr.open_dataset(ncfile)
    nt = len(nc['time'])
    f = 'E:/projects/aorc/CONUS_APCP/2008/AORC_APCP_200806.npy'
    pool = Pool(6)
    np.save(file=f,
            arr=np.transpose(np.array(list(pool.map(fun,range(0,nt))),dtype=np.float16)))
    print(time.time()-timer)

def run_in_mem(ncfile): 
    timer = time.time()
    f = 'E:/projects/hpy/examples/hydrosheds/conus_centroids.csv'
    df = pd.read_csv(f)
    df['lid'] = df['HYRIV_ID'].to_numpy(dtype=np.int32) - int(7E7)
    crd_ix = df.set_index('lid').to_xarray()
    #ncfile = 'E:/projects/aorc/CONUS_APCP/2008/AORC_APCP_200806.nc'
    nc = xr.open_dataset(ncfile)
    T = nc['time'].to_numpy()
    filename = ncfile+'.npy'
    nrow = len(df)+1
    ncol = len(T)
    output = np.memmap(filename, dtype='float16', mode='write', shape=(nrow,ncol))
    idxt=0
    for i in T:
        print(i)
        out = nc.sel(lon=crd_ix.x,lat=crd_ix.y,time= i, method='nearest')
        aux = out['APCP'].to_numpy()
        aux = np.nan_to_num(aux,copy=False,posinf=0.0,neginf=0.0)
        output[1:,idxt]=aux
        idxt+=1
    output.flush()
    print(time.time()-timer)

def get_col_row(xnc,ync,x,y):
    # Get the nc row, col for the point's lat, lon
    #need vectorized version of this function.
    col = np.argmin(np.abs(xnc - x))
    row = np.argmin(np.abs(ync - y))
    return col, row


def test_lookup_table():
    f = 'E:/projects/hpy/examples/hydrosheds/conus_centroids.csv'
    df = pd.read_csv(f)
    x = df['x'].to_list()
    y = df['y'].to_list()
    col = []
    row = []
    f = 'E:/projects/aorc/CONUS_APCP/2008/AORC_APCP_200806.nc'
    nc = Dataset(f)
    xnc = np.array(nc.variables["lon"][:])
    ync = np.array(nc.variables["lat"][:])
    nc.close()
    lid = df['HYRIV_ID'].to_numpy() - int(7E7)
    minxnc = min(xnc)
    maxxnc = max(xnc)
    minync = min(ync)
    maxync = max(ync)
    N = len(x)
    print(N)
    for i in range(0,N):
        if i % 1000==0:
            print(i)
        if x[i] < minxnc or x[i] > maxxnc or y[i] < minync or y[i] > maxync:
            col.append(-1)
            row.append(-1)
        else:
            _col,_row = get_col_row(xnc,ync,x[i],y[i])
            col.append(_col)
            row.append(_row)
    df2 = pd.DataFrame({'lid':lid,'col':col,'row':row})
    f = 'E:/projects/hpy/examples/hydrosheds/conus_lookup_aorc.csv'
    df2.to_csv(f,index=False)
    print('done')

def run2(ncfile): 
    timer = time.time()

    f = 'E:/projects/hpy/examples/hydrosheds/conus_lookup_aorc.csv'
    df = pd.read_csv(f)
    
    nc = xr.open_dataset(ncfile)
    #nc = Dataset(ncfile)
    nt = len(nc['time'])
    filename = ncfile+'.npy'
    nrow = len(df)+1
    #output = np.memmap(filename, dtype=np.float32, mode='write', shape=(nrow,nt))
    output = np.zeros( dtype=np.float32, shape=(nrow,nt))

    rows = xr.DataArray(df['row'].to_numpy(dtype=np.int32))
    cols = xr.DataArray(df['col'].to_numpy(dtype=np.int32))
    da = nc['APCP'].load()
    for t in range(0,nt):
        print(t)
        aux = da.isel(time=t,lat=rows,lon=cols).to_numpy()
        aux = np.nan_to_num(aux,copy=False,posinf=0.0,neginf=0.0)
        output[1:,t]=aux
    #output.flush()
    #np.save(file=filename,arr=output.reshape(nrow*nt))
    np.save(file=filename,arr=output)
    print(time.time()-timer)

def testzarr():
#    https://nbviewer.org/github/NOAA-OWP/AORC-jupyter-notebooks/blob/master/jupyter_notebooks/AORC_Zarr_notebook.ipynb
  #https://stackoverflow.com/questions/78537131/dealing-with-a-very-large-xarray-dataset-loading-slices-consuming-too-much-time
    import xarray as xr
    import fsspec
    import numpy as np
    import s3fs
    import zarr
    from calendar import monthrange
    base_url = f's3://noaa-nws-aorc-v1-1-1km'
    import dask

    year = '2008'
    single_year_url = f'{base_url}/{year}.zarr/'
    ds = xr.open_zarr(fsspec.get_mapper(single_year_url, anon=True), consolidated=True,chunks='auto')
    vars = ['APCP_surface',
    'DLWRF_surface',
    'DSWRF_surface',
    'PRES_surface',
    'SPFH_2maboveground',
    'TMP_2maboveground',
    'UGRD_10maboveground',
    'VGRD_10maboveground']
    var = vars[5]

    f = 'E:/projects/hpy/examples/hydrosheds/conus_centroids.csv'
    df = pd.read_csv(f)
    yy = xr.DataArray(df['y'].to_numpy(dtype=np.float32))
    xx = xr.DataArray(df['x'].to_numpy(dtype=np.float32))
    nrow = len(df)+1
    for i in range(4,13):
        dd =str(i).zfill(2)
        today1 = year+'-'+dd+'-01'
        today =pd.Timestamp(today1)
        _, num_days = monthrange(today.year, today.month)
        date_range = pd.date_range(start=today, periods=num_days*24, freq='H')
        #date_range = pd.date_range(start=today, periods=num_days, freq='D')
        nt = len(date_range)
        output = np.zeros( dtype=np.float32, shape=(nrow,nt))
        t=0
        #for tt in date_range:
        N = 24*5 #120 #5 days
        steps = int(nt / N)
        for i in range(0,steps+1):
            #0-99
            #100-199
            idx1 = N*i
            idx2 = idx1 + (N-1)
            if idx1==nt:
                break
            if idx2 >=nt:
                idx2=nt-1
            print((idx1,idx2))
            #subset = ds.sel(time=tt)
            subset = ds.sel(time=slice(date_range[idx1],date_range[idx2]))
            subset = subset[var]
            
            print(f'Variable size: {subset.nbytes/1e6:.1f} MB')
            timer = time.time()
            subset = subset.sel(latitude=yy,longitude=xx,method='nearest')
            print(time.time()-timer)
            
            timer = time.time()
            #aux = subset.values.ravel()
            subset = subset.load()
            print(time.time()-timer)
            for time_step in subset.time:
                aux = subset.sel(time=time_step)
                aux = aux.to_numpy()
                print(f'Variable size: {aux.nbytes/1e6:.1f} MB')
                aux = np.nan_to_num(aux,copy=False,posinf=0.0,neginf=0.0)
                output[1:,t]=aux
                t+=1
        filename = 'E:/projects/rush/'+var+'_'+year+dd
        np.save(file=filename,arr=output)
        #time=slice(tini,tend)


def testzarr2():
    import zarr
    import fsspec
    from calendar import monthrange
    f = 'E:/projects/hpy/examples/hydrosheds/conus_centroids.csv'
    df = pd.read_csv(f)
    yy = df['y'].to_numpy(dtype=np.float32)
    xx = df['x'].to_numpy(dtype=np.float32)
    minx = -130
    maxx = -60
    nx = 8401
    dx = (maxx - minx)/nx
    miny = 20
    maxy = 55
    ny = 4201
    dy = (maxy-miny)/ny
    ix = np.array((xx - minx )/ dx,dtype=np.int32)
    iy = np.array((yy - miny)/dy,dtype=np.int32)
    mask = np.logical_and((ix>=0),(ix<nx))
    mask = np.logical_and(mask,(iy>=0))
    mask = np.logical_and(mask, (iy<ny))
    wh = np.where(mask)
    print('stop')
    year = '2008'
    base_url = f's3://noaa-nws-aorc-v1-1-1km'
    single_year_url = f'{base_url}/{year}.zarr/'
    vars = ['APCP_surface',
    'DLWRF_surface',
    'DSWRF_surface',
    'PRES_surface',
    'SPFH_2maboveground',
    'TMP_2maboveground',
    'UGRD_10maboveground',
    'VGRD_10maboveground']
    var = vars[0]
    z1 = zarr.open(fsspec.get_mapper(single_year_url, anon=True))
    z1[var].shape #time,lat,long
    t=0
    values = z1[var][:,iy[wh],ix[wh]]
    for i in range(1,12):
        range_month = monthrange(year,i)
        for d in range_month:
            for h in range(1,24):
                pass
    print(z1.shape)
    
    
if __name__=='__main__':
    #testzarr2()
    testzarr()
    f = 'E:/projects/aorc/CONUS_APCP/2008/AORC_APCP_2008%s.nc'
    m =['01','02','03','04','05','06','07','08','09','10','11','12']
    
    #for i in range(0,len(m)):
    #    f2 = f%m[i]
    #    print(f2)
    #    run2(f2)
