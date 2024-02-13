from utils.states.states_default import get_default_states
from utils.states.states_from_nc import states_from_nc
import os
import pandas as pd

def get_states_from_manager(options=None,network:pd.DataFrame=None)->pd.DataFrame:
    """
    Load initial state data for river network.
    """
    # check if the "initial_states" key is in the "options" dictionary, if not quit the function.
    if 'initial_states' not in list(options.keys()):
        print('Error. No states option in yaml')
        quit()
    # get the file path of the key
    f = options['initial_states']
    # if f is "default", generate a default set of states based on the river network
    if f=='default':
        df = get_default_states(network)
        return df

    # if f is not "default", check if the initial state file path exists.
    if os.path.isfile(f)==False:
        print('Error. States file not found')
        quit()

    # check the extension of the initial state file
    _, extension = os.path.splitext(f)
    # If the extension is ".pkl", load it as a dataframe
    if extension =='.pkl':
        try:
            df = pd.read_pickle(f)
            return df
        except AttributeError as e:
            print('states pickle file created with different version of pandas')
            print(e)
        quit()

    # If the extension is ".csv", load it as a dataframe
    if extension == '.csv':
        try:
            df = pd.read_csv(f)
            # set the link id to be df index
            df.index = df['link_id']

            return df
        except:
            print('Unable to load the initial state file.')
        quit()


    # if the extension is ".nc", check if the "init_time" key is in the "option" dictionary
    if extension == '.nc':
        if 'init_time' not in list(options.keys()):
            print('Error. No init time option in yaml')
            quit()
        # use "init_time" to load the state from "nc" file.
        init_time = options['init_time']
        df = states_from_nc(f,init_time,network)
        return df

