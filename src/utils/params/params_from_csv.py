import pandas as pd
import numpy as np
from models.model400names import PARAM_NAMES
import sys

def params_from_csv(network:pd.DataFrame,csvfile:str):
    nlinks = network.shape[0]
    check, df = check_file(csvfile,PARAM_NAMES,network)
    df = df.astype(PARAM_NAMES)
    return df

def check_file(csvfile,PARAM_NAMES,network:pd.DataFrame):
    check=True
    df= pd.read_csv(csvfile)
    columns_model=list(PARAM_NAMES.keys())
    columns_csv = list(df.columns)
    check = np.array_equal(np.sort(columns_model),np.sort(columns_csv))
    if check==False:
        print('param names in %s do not match the model param names. please check the column names'%csvfile)
        print(columns_model)
        quit()
    
    nrows_csv = len(df)
    nrows_network = len(network)
    check = nrows_csv ==nrows_network
    if check==False:
        print('%s has %s rows and the network has %s. please check params file'%csvfile, nrows_csv,nrows_network)
        quit()
    
    df.index = df['link_id']
    rowname_csv = df.index
    rowname_network = network.index
    check = np.array_equal(np.sort(rowname_csv),np.sort(rowname_network))
    if check==False:
        print('%s has %s rows and the network has %s. please check params file'%csvfile, nrows_csv,nrows_network)
        elements_not_in_b = np.isin(rowname_csv, rowname_network, invert=True)
        print(elements_not_in_b)
        print('were not found in the parameters file')
        quit()
    
    return check, df


