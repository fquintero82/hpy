import pandas as pd
from models.model400names import PARAM_NAMES,STATES_NAMES,FORCINGS_NAMES
from utils.network.network import NETWORK_NAMES
import time as mytime
import polars as pl
from polars import col
import numpy as np



def model(df:pl.DataFrame,DT,debug=False):
    CF_MMHR_M_MIN = np.float32(1./1000.)*(1/60.) #factor .converts [mm/hr] to [m/min]
    CF_MELTFACTOR= np.float32((1/(24*60.0)) *(1/1000.0)) # mm/day/degree to m/min/degree
    CF_ET = np.float32((1e-3 / (30.0*24.0*60.0)))
    CF_METER_TO_MM = 1000
    CF_DAYS_TO_MINUTES = 24 * 60

    df = df.with_columns(
        pl.when(col('temperature')==0)
        .then(col('precipitation')* CF_MMHR_M_MIN * DT)
        .otherwise(0)
        .alias('x1')
    )
    
    df = df.with_columns(
        pl.when(col('temperature')>=col('temp_threshold'))
        .then(col('temperature')*(col('melt_factor')*CF_MELTFACTOR * DT))
        .alias('val2')
    )
    df = df.with_columns(
        pl.when(col('temperature')>=col('temp_threshold'))
        .then(pl.min_horizontal('snow','val2'))
        .alias('snowmelt')
    )
    df = df.with_columns(
        (col('snow')-col('snowmelt'))
        .alias('snow')
    )
    df = df.with_columns(
        pl.when(col('temperature')>=col('temp_threshold'))
        .then(CF_MMHR_M_MIN*DT*col('precipitation') + col('snowmelt'))
        .alias('x1') #[m]
    )

    df = df.with_columns(
        pl.when((col('temperature')!=0)&(col('temperature')< col('temp_threshold')))
        .then(col('snow') + CF_MMHR_M_MIN*DT*col('precipitation'))
        .otherwise(col('snow'))
        .alias('snow')
    )
    df = df.with_columns(
        pl.when((col('temperature')!=0)&(col('temperature')< col('temp_threshold')))
        .then(0)
        .otherwise(col('x1'))
        .alias('x1')
    )
    df = df.with_columns(
        (col('precipitation')*col('area_hillslope'))
        .alias('basin_precipitation')
    )
    df = df.with_columns(
        (CF_METER_TO_MM*col('snow')*col('area_hillslope'))
        .alias('basin_swe')
    )

    df = df.with_columns(
        pl.max_horizontal(0,(col('x1') + col('static') - col('max_storage')/1000.))
        .alias('x2')
    )

    df = df.with_columns(
        pl.when(col('frozen_ground')==1)
        .then(col('x1'))
        .otherwise(col('x2'))
        .alias('x2')
    )
    # print((df['x2']).describe()[2])

    df = df.with_columns(
        pl.min_horizontal(
            (col('evapotranspiration')*CF_ET*DT),
            'static'
            )
        .alias('out1')
    )
    df = df.with_columns(
        (col('static') + col('x1') - col('x2')-col('out1'))
        .alias('static')
    )
    # print((df['static']+df['x1']-df['x2']-df['out1']).describe()[2])


    df = df.with_columns(
        (CF_METER_TO_MM*col('out1')*col('area_hillslope'))
        .alias('basin_evapotranspiration')
    )
    df = df.with_columns(
        (CF_METER_TO_MM*col('static')*col('area_hillslope'))
        .alias('basin_static')
    )

    df = df.with_columns(
        (col('infiltration') * CF_MMHR_M_MIN * DT)
        .alias('infiltration')
    )
    df = df.with_columns(
        pl.when(col('frozen_ground')==1)
        .then(0)
        .otherwise(col('infiltration'))
        .alias('infiltration')
    )

    df = df.with_columns(
        pl.min_horizontal('x2','infiltration')
        .alias('x3')
    )
    df = df.with_columns(
        (col('surface_velocity') * 60* col('channel_length') / col('area_hillslope'))
        .alias('w')
    )

    df = df.with_columns(
        pl.min_horizontal(1,'w')
        .alias('w')
    )
    df = df.with_columns(
        pl.min_horizontal(
            (col('surface')*col('w')*DT),
            'surface'
            )
        .alias('out2')
    )

    df = df.with_columns(
        (col('surface')+col('x2')-col('x3')-col('out2'))
        .alias('surface')
    )
    df = df.with_columns(
        (CF_METER_TO_MM*col('out2') *col('area_hillslope'))
        .alias('basin_surface')
    )
    df = df.with_columns(
        (col('percolation') * CF_MMHR_M_MIN * DT)
        .alias('percolation')
    )
    df=df.with_columns(
        pl.min_horizontal('x3','percolation')
        .alias('x4')
    )

    df= df.with_columns(
        (DT * col('subsurface') / (col('tr_subsurface')* CF_DAYS_TO_MINUTES))
        .alias('out3')
    )
    df= df.with_columns(
        (col('subsurface')+col('x3')-col('x4')-col('out3'))
        .alias('subsurface')
    )

    df= df.with_columns(
        (CF_METER_TO_MM*col('out3') *col('area_hillslope'))
        .alias('basin_subsurface')
    )
    df= df.with_columns(
        (DT * col('groundwater') / (col('tr_groundwater')* CF_DAYS_TO_MINUTES))
        .alias('out4')
    )

    df= df.with_columns(
        (col('groundwater')+col('x4')-col('out4'))
        .alias('groundwater')
    )
    df= df.with_columns(
        (CF_METER_TO_MM*col('out4') *col('area_hillslope'))
        .alias('basin_groundwater')
    )
    df= df.with_columns(
        (col('volume')+ (col('out4')+col('out3')+col('out2')) *col('area_hillslope'))
        .alias('volume')
    )

    return df

def transfer_df(states:pd.DataFrame,df):
    col = states.columns
    for i in range(len(col)):
        states[col[i]]=np.array(df[col[i]].to_numpy(),dtype=STATES_NAMES[col[i]])


def runoff1(states:pd.DataFrame,
    forcings:pd.DataFrame,
    params:pd.DataFrame,
    network:pd.DataFrame,
    time_step_sec:int):
    
    t1 = mytime.time()
    DT = time_step_sec / 60. #minutes
    if check_input_names(states=states,forcings=forcings,params=params,network=network)==False:
        return
    if check_input_values(states=states,forcings=forcings,params=params,network=network,DT=DT)==False:
        return
    
    
    a = pl.from_pandas(states)
    n = len(forcings.columns)
    b = pl.from_pandas(forcings.iloc[:,1:n])
    n = len(params.columns)
    c = pl.from_pandas(params.iloc[:,1:n])
    d = pl.from_pandas(network[['channel_length','area_hillslope']])
    df = pl.concat([a,b,c,d],how='horizontal')
    # if (states.index==forcings.index).all()==False:
    #     print('different order')
    #     quit()
    df = model(df,DT,debug=True)
    transfer_df(states,df)
    x = int(1000*(mytime.time()-t1))
    print('completed runoff in {x} msec'.format(x=x))


def check_input_names(states:pd.DataFrame,
    forcings:pd.DataFrame,
    params:pd.DataFrame,
    network:pd.DataFrame):
    flag = True

    for i in STATES_NAMES:
        if(i not in states.columns):
            flag = False
            print('column {} in dataframe state was not found'.format(i))
            return flag
    
    for i in PARAM_NAMES:
        if(i not in params.columns):
            flag = False
            print('column {} in dataframe params was not found'.format(i))
            return flag
    
    for i in FORCINGS_NAMES:
        if(i not in forcings.columns):
            flag = False
            print('column {} in dataframe forcings was not found'.format(i))
            return flag
    
    for i in NETWORK_NAMES:
        if(i not in network.columns):
            flag = False
            print('column {} in dataframe network was not found'.format(i))
            return flag
    
    return flag

def check_input_values(states:pd.DataFrame,
    forcings:pd.DataFrame,
    params:pd.DataFrame,
    network:pd.DataFrame,
    DT:int):
    flag=True
    
    flag = (DT==0)
    if flag==True:
        print("Error DT is zero")
        return False
    
    flag = network['channel_length'].to_numpy().all()
    if flag==False:
        print("Error Parameter channel_length has zeros")
        return False
    
    flag = network['area_hillslope'].to_numpy().all()
    if flag==False:
        print("Error Parameter area_hillslope has zeros")
        return False
    
    flag = params['tr_subsurface'].to_numpy().all()
    if flag==False:
        print("Error Parameter tr_subsurface has zeros")
        return False
    
    flag = params['tr_groundwater'].to_numpy().all()
    if flag==False:
        print("Error Parameter tr_groundwater has zeros")
        return False
    
    flag = states.index.to_numpy().all()
    if flag==False:
        print("Error States cannot have linkid index zero")
        return False
    
    flag = np.array(states['static'].to_numpy()>=0.5).all()
    if flag==True:
        print("static storage cannot be larger than 0.5m")
        quit()
    
    flag = np.array(states['static'].to_numpy()<0).all()
    if flag==True:
        print("static storage cannot be negative")
        quit()

    flag = np.array(states['surface'].to_numpy()<0).all()
    if flag==True:
        print("surface storage cannot be negative")
        quit()

    flag = np.array(states['subsurface'].to_numpy()<0).all()
    if flag==True:
        print("subsurface storage cannot be negative")
        quit()

    flag = np.array(states['groundwater'].to_numpy()<0).all()
    if flag==True:
        print("groundwater storage cannot be negative")
        quit()

    flag = np.array(states['volume'].to_numpy()<0).all()
    if flag==True:
        print("volume storage cannot be negative")
        quit()

    flag = np.array(states['discharge'].to_numpy()<0).all()
    if flag==True:
        print("discharge storage cannot be negative")
        quit()

    flag = params.index.to_numpy().all()
    if flag==False:
        print("Error Params cannot have linkid index zero")
        return False
    
    flag = forcings.index.to_numpy().all()
    if flag==False:
        print("Error Forcings cannot have linkid index zero")
        return False
    
    flag = network.index.to_numpy().all()
    if flag==False:
        print("Error Network cannot have linkid index zero")
        return False
    
    return flag



