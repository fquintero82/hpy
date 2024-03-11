from utils.network.network import combine_rvr_prm,network_from_rvr_file,get_idx_up_down
from utils.network.network import get_default_network
from utils.network.network_symbolic import process_all
import pandas as pd

def test1():
    rvr_file ='examples/cedarrapids1/367813.rvr'
    #df = network_from_rvr_file(rvr_file)
    prm_file ='examples/cedarrapids1/367813.prm'
    df = combine_rvr_prm(prm_file,rvr_file)
    df.to_pickle('examples/cedarrapids1/367813.pkl')



def test2():
    rvr_file ='examples/small/small.rvr'
    prm_file ='examples/small/small.prm'
    df = combine_rvr_prm(prm_file,rvr_file)
    df.to_pickle('examples/small/small.pkl')

def test3():
    rvr_file ='examples/hydrosheds/conus.rvr'
    prm_file ='examples/hydrosheds/conus.prm'
    df = combine_rvr_prm(prm_file,rvr_file)
    df.to_pickle('examples/hydrosheds/conus.pkl')

def test4():
    rvr_file ='examples/hydrosheds/conus.rvr'
    df = network_from_rvr_file(rvr_file)
    get_idx_up_down(df)

def test5():
    rvr_file ='/Users/felipe/tmp/iowa/topo.rvr'
    prm_file ='/Users/felipe/tmp/iowa/params.prm'
    df = combine_rvr_prm(prm_file,rvr_file)
    process_all(df)
    df.to_pickle('/Users/felipe/tmp/iowa/iowa_network.pkl')

def test6():
    f ='examples/cedarrapids1/367813.pkl'
    df = pd.read_pickle(f)
    process_all(df)
    df.to_pickle('/Users/felipe/tmp/367813_imac.pkl')

def test7():
    f ='E:/projects/hpy/examples/cedarrapids1/367813.pkl'
    df = pd.read_pickle(f)
    process_all(df)
    df.to_pickle('/Users/felipe/tmp/367813_imac.pkl')

def test8():
    f ='examples/hydrosheds/conus.pkl'
    df = pd.read_pickle(f)
    process_all(df)
    df.to_pickle('/Users/felipe/tmp/conus_imac.pkl')

def test9():
    rvr= '/Users/felipe/tmp/iowa_operational/ifis_iowa.rvr'
    prm= '/Users/felipe/tmp/iowa_operational/ifis_iowa.prm'
    df = combine_rvr_prm(prm,rvr,paramsplit=True)

def test10():
    rvr = 'C:/Users/fquinteroduque/Desktop/tmp/ifisff/topo.rvr'
    prm = 'C:/Users/fquinteroduque/Desktop/tmp/ifisff/params.prm'
    df = combine_rvr_prm(prm,rvr,paramsplit=False)
    df.to_pickle('E:/projects/et/hydrology/iowa_win.pkl')

def test11():
    rvr_file ='examples/miss/miss.rvr'
    prm_file ='examples/miss/miss.prm'
    df = combine_rvr_prm(prm_file,rvr_file)
    df.to_pickle('examples/miss/miss.pkl')

def test12():
    f = '/Users/felipe/tmp/367813_imac.pkl'
    f2 = '/Users/felipe/tmp/367813_imac_fullrouting.pkl'
    df = pd.read_pickle(f)
    process_all(df)
    df.to_pickle(f2)

def create_binary_network():
    import numpy as np
    count_done =0
    count_links = 0
    n=2**9
    n1 = 2**10
    arr = np.zeros(shape=(n1+1,3),dtype=np.uint)
    while count_done<n:
        arr[count_done,0] = count_done
        arr[count_done,1] = count_links+1
        arr[count_done,2] = count_links+2
        count_done +=1
        count_links+=2
    a = np.arange(n,n1+1)
    arr[n:(n1-1),0]=a

if __name__ == "__main__":
    test3()
    # f ='/Users/felipe/tmp/iowa/iowa_network.pkl'
    # df = pd.read_pickle(f)