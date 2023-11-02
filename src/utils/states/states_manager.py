from utils.states.states_default import get_default_states
import os
import pandas as pd

def get_states_from_manager(options=None,network:pd.DataFrame=None):
    if 'states' not in list(options.keys()):
        print('Error. No states option in yaml')
        quit()
    f = options['states']
    if f=='default':
        df = get_default_states(network)
        return df
    if os.path.isfile(f)==False:
        print('Error. States file not found')
        quit()
    _, extension = os.path.splitext(f)
    if extension =='.pkl':
        try:
            df = pd.read_pickle(f)
            return df
        except AttributeError as e:
            print('states pickle file created with different version of pandas')
            print(e)
        quit()
    if extension == '.csv':
        pass   
