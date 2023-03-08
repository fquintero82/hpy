import pandas as pd

def save_to_netcdf(states:pd.DataFrame,time:int):
    pass

def save_to_pickle(states:pd.DataFrame,time:int):
    f = '../examples/cedarrapids1/out/{}.pkl'.format(time)
    states['discharge'].to_pickle(f)

def read_from_pickle(states:pd.DataFrame,fileinput:str):
    pass
