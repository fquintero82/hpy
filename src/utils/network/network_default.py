import pandas as pd
from model400names import STATES_NAMES
import numpy as np

def get_default_network():
    f = 'examples/cedarrapids1/367813_network.pkl'
    df = pd.read_pickle(f)
    return df

