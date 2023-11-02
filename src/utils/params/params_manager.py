import os
from os.path import splitext
import pandas as pd
from utils.params.params_from_csv import params_from_csv
from utils.params.params_default import get_default_params
def get_params_from_manager(options=None,network:pd.DataFrame=None):
    if 'parameters' not in list(options.keys()):
        print('Error. No parameters option in yaml')
        quit()
    f = options['parameters']
    if f=='default':
        df = get_default_params(network)
        return df
    
    if os.path.isfile(f)==False:
        print('Error. Parameters file not found')
        quit()
    _, extension = os.path.splitext(f)
    if extension =='.pkl':
        try:
            df = pd.read_pickle(f)
            return df
        except AttributeError as e:
            print('parameters pickle file created with different version of pandas')
            print(e)
        quit()
    if extension == '.csv':
        df = params_from_csv(f)
        return df
    
