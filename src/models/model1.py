import numpy as np
import pandas as pd


PARAM_NAMES = ['link_id']

STATES_NAMES = {'link_id':np.uint32,
                'runoff':np.float16,
                'mean_areal_runoff':np.float16
                }
FORCINGS_NAMES=['link_id','runoff']

CF_NAMES ={
    'forcings.runoff':'surface_runoff_amount',
    'states.runoff':'surface_runoff_amount'
}

VAR_TYPES ={
    'forcings.runoff':'float',
    'states.runoff':'float'
}

CF_UNITS={
    'forcings.runoff':'mm',
    'states.volume':'m3',
    'states.mean_areal_runoff':'mm'

}

CF_LOCATION={
    'forcings.runoff':'face',
    'states.runoff':'face'
}

def runoff1(
    states:pd.DataFrame,
    forcings:pd.DataFrame,
    params:pd.DataFrame,
    network:pd.DataFrame,
    DT:int):
    CF1 = np.float16(1./1000.)
    states['volume'] = forcings['runoff'] * network['area_hillslope'] * CF1 # [mm] to m to m3
