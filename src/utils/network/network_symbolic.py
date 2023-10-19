import numpy as np
import pandas as pd
from utils.network.network import get_default_network, update_network_pickle,get_network_from_file
import sys
from math import factorial
import numba
from multiprocessing import Pool, Manager, Process,freeze_support
import time
import pickle
from hlm import HLM
import warnings
from numba import njit,prange
# https://stackoverflow.com/questions/69423036/use-eval-in-numba-numbalsoda#:~:text=eval%20is%20fundamentally%20incompatible%20with,calls)%20to%20be%20strongly%20typed.
def test_eval():
    var = 'x[0] + x[1]'
    x = np.array([1,1])
    # eval(var)
    var='0'
    N = int(1e5) #este es el maximo que da mi portatil 100,000
    print(N)
    sys.setrecursionlimit(N)
    for i in np.arange(N):
        var+= '+ x[%d]'%i
    x=np.ones(N)
    eval(var)

def _process_unit(idx:np.int32,
                    idx_upstream_links:np.ndarray,
                    order:np.int32,
                    expr:list):
        if order < 10:
            myexpr = '1/float(factorial({order}-1)) * P[{idx}-1]**({order}-1) * X[{idx}-1] * T**({order}-1) * 2.718 ** (-P[{idx}-1] * T)'
            myexpr = myexpr.format(order=order,idx=idx)
            expr.append(myexpr)
            myidx_upstream_links = idx_upstream_links[idx - 1]
            if (myidx_upstream_links!=0).any():
                for new_idx in myidx_upstream_links:
                    _process_unit(new_idx,idx_upstream_links,order+1,expr)

def process_unit(idx:np.int32,
                idx_upstream_links:np.ndarray):
    expr = []
    order=1
    _process_unit(idx,idx_upstream_links,order,expr)
    expr = '+'.join(expr)
    return expr

# def process_unit(x:list):
#     idx = x[1]
#     idx_upstream_links = x[2]
#     process_unit(idx,idx_upstream_links)
    

def test_process_unit():
    network = get_default_network()
    N = len(network)
    link_id = 367813
    idx = 32715
    idx = 40163
    idx_upstream_links = network['idx_upstream_link'].to_numpy()
    out = process_unit(idx,idx_upstream_links)

def process_all_map(network:pd.DataFrame):
    N = len(network)
    idx_upstream_links = network['idx_upstream_link'].to_numpy()
    idxs = network['idx'].to_numpy()
    out = list(map(process_unit,idxs[0:9],idx_upstream_links))

# @njit(parallel=True)
# def eval_numba(expr,X,P,T):
#     N = len(expr)
#     out =np.zeros(shape=(N,))
#     X=X
#     P=P
#     T=T
#     for i in prange(N):
#         out[i] = eval(expr[i])
#     return out

def eval1(x):
    globals()
    locals()
    return eval(x)

def test_eval():
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    instance.init_from_file(config_file,option_solver=False)
 
    N=len(instance.network)
    initial_state = np.ones(shape=(N,))
    expr = instance.network['expression'].to_numpy()
    P = (instance.params['river_velocity'] / instance.network['channel_length']).to_numpy()
    d = {'X':initial_state,
        'P':P,
        'T':instance.time_step_sec,
        'factorial':factorial}
    T=instance.time_step_sec
    X = np.ones(shape=(N,))
    out = np.zeros(shape=(N,))
    t=time.time()
    # for i in np.arange(N):
    #     # print(i)
    #     out[i] = eval(expr[i]) #no usar diccionario porque se queja del factorial
    # print('done in %f sec'%(time.time()-t))
    globs = globals()
    locs = locals()
    # t=time.time()
    # out = [eval(expr[i],globs,locs) for i in np.arange(N)]
    # print('done in %f sec'%(time.time()-t))
    # t=time.time()
    # i = np.arange(N)
    # out = list(map(lambda i: eval(expr[i],d),i))
    # print('done in %f sec'%(time.time()-t))

    t=time.time()
    with Pool() as pool:
        out = pool.map(eval1,expr)
    print('done in %f sec'%(time.time()-t))
    
 

def process_all():
    network = get_default_network()
    N = len(network)
    idx_upstream_links = network['idx_upstream_link'].to_numpy()
    idxs = network['idx'].to_numpy()
    expression=np.empty(shape=(N,),dtype=object)

    for i in np.arange(N):
        if i % 10000 ==0:
            print('row %s'%i)
        out=process_unit(idxs[i],idx_upstream_links)
        expression[i] = out
    network['expression'] = expression
    f = 'examples/cedarrapids1/367813.pkl'
    update_network_pickle(network,f)

def process_all_map(network:pd.DataFrame):
    N = len(network)
    idx_upstream_links = network['idx_upstream_link'].to_numpy()
    idxs = network['idx'].to_numpy()
    expression=np.chararray(shape=(N,))
# def process_all_multiprocessing(network:pd.DataFrame):
#     N = len(network)
#     _idx_upstream_links = network['idx_upstream_link'].to_numpy()
#     _idxs = network['idx'].to_numpy()
#     ncpu = 2
#     manager = Manager()
#     idx_upstream_link = manager.list(_idx_upstream_links)
#     idx = manager.list(_idxs)
#     _expr = [[] for x in range(4)]
#     expr = manager.list(_expr)
#     jobs = []
#     for idx in _idxs[0:3]:
#         order=1
#         j = Process(target=process_unit,args=(idx,idx_upstream_link,order,expr[idx]))
#         jobs.append(j)
#     for j in jobs:
#         j.start()
#     for j in jobs:
#         j.join()
#     print('stop')

# def test_process_all_multiprocessing():
#     freeze_support()
#     t = time.time()
#     network = get_default_network()
#     process_all_multiprocessing(network)
#     print(time.time()-t)

# @numba.jit(target='cuda')
# def process_all_cuda(idx_upstream_links:np.ndarray,idxs:np.ndarray):
#     N = len(idx_upstream_links)
#     out = np.empty(shape=(N,),dtype=object)
#     for i in np.arange(N):
#         order = 1
#         expr = []
#         process_unit(idxs[i],idx_upstream_links,order,expr)
#         out[i]=np.array(expr)
#     return out

# @numba.jit(target='cuda')
# def testcuda1():
#     N=1e6
#     a=0
#     for i in range(N):
#         a+=i
#     print(a)
if __name__ == '__main__':
    # process_all()
    test_eval()
    # test_process_all_multiprocessing()
    # test_process_unit()