import pandas as pd
import numpy as np
import datetime as dt

file = '/Users/felipe/hio/hio/examples/timeseries1/timeseries_et.csv'
file = 'E:\projects\iowa_operational\IA0000_2023.txt'
def get_values(time:int,options=None):
    if options is not None:
        file = options['path']
    data = pd.read_csv(file,parse_dates=True,
                       names=['station','stationname','date','doy','highc','lowc'],
                       header=0,
                       dtype={'station':str,
                              'stationname':str,
                              'date':str,
                              'doy':np.int,
                              'highc':np.float32,
                              'lowc':np.float32})
    unixtime =(pd.to_datetime(data['date']) - dt.datetime(1970,1,1)).dt.total_seconds()
    val=np.interp(time,unixtime,
                  (data['highc']-data['lowc'])/2.0)
    lid=None
    if val is None:
        return lid,0
    if np.isnan(val):
        return lid,0
    
    return lid,val 
