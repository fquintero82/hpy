import pandas as pd
import numpy as np
import datetime as dt

file = '/Users/felipe/hio/hio/examples/timeseries1/timeseries_et.csv'
def get_values(time:int,options=None):
    if options is not None:
        file = options['path']
    data = pd.read_csv(file,parse_dates=True,names=['date','val'],header=0,dtype={'date':str,'val':np.float16})
    unixtime =(pd.to_datetime(data['date']) - dt.datetime(1970,1,1)).dt.total_seconds()
    val=np.interp(time,unixtime,data['val'])
    lid=None
    return lid,val 
