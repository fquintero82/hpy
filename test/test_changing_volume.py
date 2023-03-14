from routing import linear_velocity
from utils.network.network import combine_rvr_prm
from test_dataframes import getDF_by_size
from model400names import STATES_NAMES
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils.params.params_default import get_default_params
from utils.states.states_default import get_default_states


def test1():
    network = pd.read_pickle('../examples/cedarrapids1/367813_network.pkl')
    nlinks = network.shape[0]
    states = get_default_states(network)
    params = get_default_params(network)
    states['depth'] = 0
    params['width'] = 1
    states['river_velocity'] = 0
    