from utils.network.network import combine_rvr_prm,network_from_rvr_file,get_idx_up_down
from utils.network.network import get_default_network
from utils.network.network_symbolic import process_all
import pandas as pd

def test1():
    rvr_file ='examples/cedarrapids1/367813.rvr'
    #df = network_from_rvr_file(rvr_file)
    prm_file ='examples/cedarrapids1/367813.prm'
    df = combine_rvr_prm(prm_file,rvr_file)
    df.to_pickle('examples/cedarrapids1/367813_network.pkl')



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
    df.to_pickle('/Users/felipe/tmp/conus_macbook.pkl')

# def testadjmat():
#     network = get_default_network()
#     A = get_adjacency_matrix(network,False)
#     file1 = open('examples/cedarrapids1/367813_adj.pkl','wb')
#     #file2 = open('examples/cedarrapids1/367813_adj.np','wb')
#     pickle.dump(A,file1)
#  #   np.save(file2,A) #same size as pkl
#     file1.close()
# #    file2.close()

if __name__ == "__main__":
    test6()
    # f ='/Users/felipe/tmp/iowa/iowa_network.pkl'
    # df = pd.read_pickle(f)