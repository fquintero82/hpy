import pandas as pd
import numpy as np
from model400names import PARAM_NAMES
import sys

def get_params(network:pd.DataFrame,csvfile:str):
    nlinks = network.shape[0]
    csvfile = 'examples/cedarrapids1/parameters.csv'
    check, dfcsv = check
    df = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(PARAM_NAMES))),
        columns=list(PARAM_NAMES.keys()))
    df = df.astype(PARAM_NAMES)
    aux = list(PARAM_NAMES.keys())
    for ii in range(len(PARAM_NAMES)):
        df.loc[:,aux[ii]] = np.array(PARAM_DEFAULT_VALUES[aux[ii]],dtype=PARAM_NAMES[aux[ii]])

    df['link_id'] = network['link_id'].to_numpy()
    df.index = network['link_id'].to_numpy()
    return df

def check_file(csvfile,PARAM_NAMES,network:pd.DataFrame):
    check=True
    dfcsv= pd.read_csv(csvfile)
    columns_model=list(PARAM_NAMES.keys())
    columns_csv = list(dfcsv.columns)
    check = np.array_equal(np.sort(columns_model),np.sort(columns_csv))
    if check==False:
        print('column names in %s do not match the model column names. please check the column names'%csvfile)
        print(columns_model)
        quit()
    
    nrows_csv = len(dfcsv)
    nrows_network = len(network)
    check = nrows_csv ==nrows_network
    if check==False:
        print('%s has %s rows and the network has %s. please check params file'%csvfile, nrows_csv,nrows_network)
        quit()
    
    dfcsv.index = dfcsv['link_id']
    rowname_csv = dfcsv.index
    rowname_network = network.index
    check = np.array_equal(np.sort(rowname_csv),np.sort(rowname_network))
    if check==False:
        print('%s has %s rows and the network has %s. please check params file'%csvfile, nrows_csv,nrows_network)
        elements_not_in_b = np.isin(rowname_csv, rowname_network, invert=True)

        quit()
    
    return check, dfcsv


