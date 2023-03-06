import numpy as np

PARAM_NAMES = ['link_id','river_velocity','lambda1','lambda2','max_storage','infiltration',
            'percolation','surface_velocity','alfa3','alfa4','temp_threshold','melt_factor']
NETWORK_NAMES ={'link_id':np.uint32,
                'downstream_link':np.uint32,
                'idx_downstream_link':np.uint32,
                'upstream_link':np.uint32,
                'idx_upstream_link':object,
                'channel_length':np.float16,
                'area_hillslope':np.float16,
                'drainage_area':np.float16,
                }

STATES_NAMES = ['link_id','snow','static','surface','subsurface','groundwater','discharge']
FORCINGS_NAMES=['link_id','precipitation','evapotranspiration','temperature','frozen_ground','discharge']

CF_NAMES ={
    'params.river_velocity':'params.river_velocity',
    'params.lambda1':'params.lambda1',
    'params.lambda2':'params.lambda2',
    'params.max_storage':'params.max_storage',
    'params.infiltration':'params.infiltration',
    'params.percolation':'params.percolation',
    'params.alfa3':'params.alfa3',
    'params.alfa4':'params.alfa4',
    'params.temp_threshold':'params.temp_threshold',
    'params.melt_factor':'params.melt_factor',
    'forcings.precipitation':'precipitation_flux',
    'forcings.evapotranspiration':'water_evapotranspiration_flux',
    'forcings.temperature':'air_temperature',
    'forcings.frozen_ground':'soil_temperature',
    'forcings.discharge':'water_volume_transport_into_sea_water_from_rivers',
    'states.snow':'surface_snow_amount',
    'states.static':'',
    'states.surface':'surface_runoff_amount',
    'states.subsurface':'subsurface_runoff_amount',
    'states.groundwater':'baseflow_amount',
    'states.discharge':'water_volume_transport_into_sea_water_from_rivers'
}

VAR_TYPES ={
    'params.river_velocity':'float',
    'params.lambda1':'float',
    'params.lambda2':'float',
    'params.max_storage':'float',
    'params.infiltration':'float',
    'params.percolation':'float',
    'params.alfa3':'float',
    'params.alfa4':'float',
    'params.temp_threshold':'float',
    'params.melt_factor':'float',
    'forcings.precipitation':'float',
    'forcings.evapotranspiration':'float',
    'forcings.temperature':'float',
    'forcings.frozen_ground':'int',
    'forcings.discharge':'float',
    'states.snow':'float',
    'states.static':'float',
    'states.surface':'float',
    'states.subsurface':'float',
    'states.groundwater':'float',
    'states.discharge':'float'
}

CF_UNITS={
    'params.river_velocity':'m s-1',
    'params.lambda1':'',
    'params.lambda2':'',
    'params.max_storage':'mm',
    'params.infiltration':'mm h-1',
    'params.percolation':'mm h-1',
    'params.alfa3':'',
    'params.alfa4':'',
    'params.temp_threshold':'C',
    'params.melt_factor':'mm day-1 C-1',
    'forcings.precipitation':'mm h-1',
    'forcings.evapotranspiration':'mm h-1',
    'forcings.temperature':'C',
    'forcings.frozen_ground':'',
    'forcings.discharge':'m3 s-1',
    'states.snow':'m',
    'states.static':'m',
    'states.surface':'m',
    'states.subsurface':'m',
    'states.groundwater':'m',
    'states.discharge':'m3 s-1'
}

CF_LOCATION={
    'params.river_velocity':'edge',
    'params.lambda1':'face',
    'params.lambda2':'face',
    'params.max_storage':'face',
    'params.infiltration':'face',
    'params.percolation':'face',
    'params.alfa3':'face',
    'params.alfa4':'face',
    'params.temp_threshold':'face',
    'params.melt_factor':'face',
    'forcings.precipitation':'face',
    'forcings.evapotranspiration':'face',
    'forcings.temperature':'face',
    'forcings.frozen_ground':'face',
    'forcings.discharge':'edge',
    'states.snow':'face',
    'states.static':'face',
    'states.surface':'face',
    'states.subsurface':'face',
    'states.groundwater':'face',
    'states.discharge':'edge'
}