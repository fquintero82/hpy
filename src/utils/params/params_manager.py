import os
from os.path import splitext
import pandas as pd
from utils.params.params_from_csv import params_from_csv
from utils.params.params_default import get_default_params


def get_params_from_manager(options=None, network: pd.DataFrame = None):
    """
    Load model parameters for each hillslope.
    """
    # check if "parameter" is in the "option" dictionary
    if 'parameters' not in list(options.keys()):
        print('Error. No parameters option in yaml')
        quit()

    f = options['parameters']
    # if f is "default", load default set of parameter for the river network
    if f == 'default':
        df = get_default_params(network)
        return df

    # if f is not "default", check if parameter exists.
    if os.path.isfile(f) == False:
        print('Error. Parameters file not found')
        quit()

    # get the extension of the file
    _, extension = os.path.splitext(f)

    # if the extension is ".pkl", then load the dataframe
    if extension == '.pkl':
        try:
            df = pd.read_pickle(f)
            return df
        except AttributeError as e:
            print('parameters pickle file created with different version of pandas')
            print(e)
        quit()

    # if the extension is ".csv", then load the dataframe
    if extension == '.csv':
        # df = params_from_csv(network, f)
        df = pd.read_csv(f)
        # set the lind ids as index
        df.index = df['link_id']

        return df

    print('no option found for parameters')
    quit()
    
