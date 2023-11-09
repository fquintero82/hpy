from hlm import HLM
import polars as pl
from polars import col
import numpy as np

CF_MMHR_M_MIN = np.float32(1./1000.)*(1/60.) #factor .converts [mm/hr] to [m/min]
CF_MELTFACTOR= np.float32((1/(24*60.0)) *(1/1000.0)) # mm/day/degree to m/min/degree
CF_ET = np.float32((1e-3 / (30.0*24.0*60.0)))
CF_METER_TO_MM = 1000
DT = 3600
CF_DAYS_TO_MINUTES = 24 * 60

def model(df:pl.DataFrame):

    df = df.with_columns(
        pl.min_horizontal(col("precipitation")+1,col('precipitation')-1)
        .alias('test')
    )

    df = df.with_columns(
        pl.when(col('temperature')==0)
        .then(col('precipitation')* CF_MMHR_M_MIN * DT)
        .otherwise(0)
        .alias('x1')
    )
    df = df.with_columns(
        pl.when(col('temperature')>=col('temp_threshold'))
        .then(col('snow'))
        .alias('val1')
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
        .alias('x1')
    )
    df = df.with_columns(
        pl.when(col('temperature')!=0 & col('temperature')< col('temp_threshold'))
        .then(col('snow') + CF_MMHR_M_MIN*DT*col('precipitation'))
        .alias('snow')
    )
    df = df.with_columns(
        pl.when(col('temperature')!=0 & col('temperature')< col('temp_threshold'))
        .then(0)
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
        (pl.lit(0))
        .alias('val1')
    )
    df = df.with_columns(
        (col('x1') + col('static') - col('max_storage')/1000)
        .alias('val2')
    )
    df = df.with_columns(
        pl.when(col('val1')<=col('val2'))
        .then(col('val2'))
        .otherwise(col('val1'))
        .alias('x2')
    )
    df = df.with_columns(
        pl.when(col('frozen_ground')==1)
        .then(col('x1'))
        .otherwise(col('x2'))
        .alias('x2')
    )
    df = df.with_columns(
        (col('evapotranspiration')*CF_ET*DT)
        .alias('val1')
    )
    df = df.with_columns(
        (col('static'))
        .alias('val2')
    )

    df = df.with_columns(
        pl.min_horizontal('val1','val1')
        .alias('out1')
    )
    df = df.with_columns(
        (col('static') + col('x1') - col('x2')-col('out1'))
        .alias('static')
    )

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
        (col('out4') *col('area_hillslope'))
        .alias('volume')
    )

instance= HLM()

config_file = 'examples/cedarrapids1/cedar_example.yaml'
instance.init_from_file(config_file,option_solver=False)

a = pl.from_pandas(instance.states)
b = pl.from_pandas(instance.forcings.iloc[:,1:-1])
c = pl.from_pandas(instance.params.iloc[:,1:-1])
d = pl.from_pandas(instance.network[['channel_length','area_hillslope']])
N = len(instance.network)
DT = instance.time_step_sec
df = pl.concat([a,b,c,d],how='horizontal')
model(df)
