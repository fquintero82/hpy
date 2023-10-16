import numpy as np
import pandas as pd
from utils.network.network import get_default_network
import sys
from math import factorial
import numba
from multiprocessing import Pool, Manager, Process,freeze_support
import time

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


# def process_unit(idx:np.int32,
#                  idx_upstream_links:np.ndarray,
#                  order:np.int32,
#                  expr:list)->str:
#     # expr = 'X[{idx}] * 2.718 ** (-P[{idx}] * T)'
#     myexpr = '1/factorial({order}-1) * P**({order}-1) * X[{idx}] * T**({order}-1) * 2.718 ** (-P[{idx}] * T)'
#     # myexpr = 'X[{idx}] * P[{idx}] * 2.718 ** (-P[{idx}] * T) * ({order} * 1/factorial({order}-1) * (T * P[{idx}])**({order}-1)
    
#     myexpr = myexpr.format(order=order,idx=idx)
#     # expr = expr + myexpr
#     expr.append(myexpr)
#     # print(order)
#     # print(expr)
#     myidx_upstream_links = idx_upstream_links[idx - 1]
#     if (myidx_upstream_links!=0).any():
#         for new_idx in myidx_upstream_links:
#             process_unit(new_idx,idx_upstream_links,order+1,expr)
#     return expr

def process_unit(idx:np.int32,
                 idx_upstream_links:np.ndarray,
                 order:np.int32,
                 expr:list):
    if idx % 10000 ==0:
        print('row %s'%idx)

    myexpr = '1/factorial({order}-1) * P**({order}-1) * X[{idx}] * T**({order}-1) * 2.718 ** (-P[{idx}] * T)'
    myexpr = myexpr.format(order=order,idx=idx)
    # expr = expr + myexpr
    expr.append(myexpr)
    # expr[oldidx].append(myexpr)
    # expr = np.append(expr,myexpr)
    myidx_upstream_links = idx_upstream_links[idx - 1]
    if (myidx_upstream_links!=0).any():
        for new_idx in myidx_upstream_links:
            process_unit(new_idx,idx_upstream_links,order+1,expr)

def test_process_unit():
    network = get_default_network()
    N = len(network)
    link_id = 367813
    idx = 32715
    order = 1
    idx_upstream_links = network['idx_upstream_link'].to_numpy()
    # expr = np.empty(shape=(N,),dtype=object)
    expr = [[] for x in range(N)]
    process_unit(idx,idx_upstream_links,order,expr[idx])
    print(len(expr))
    X = np.ones(N)
    T = 1
    P = np.ones(N)
    out = eval(expr[1])

def process_all(network:pd.DataFrame):
    N = len(network)
    df = pd.DataFrame({'formula':np.zeros(shape=(N,))},dtype=object)
    out = np.zeros(shape=(N,),dtype=object)
    df.index = network.index
    idx_upstream_links = network['idx_upstream_link'].to_numpy()
    idxs = network['idx'].to_numpy()
    for i in np.arange(N):
        print(i)
        order = 1
        expr = [[] for x in range(N)]
        process_unit(idxs[i],idx_upstream_links,order,expr[idxs[i]])
        out[i]=np.array(expr)

def process_all_multiprocessing(network:pd.DataFrame):
    N = len(network)
    _idx_upstream_links = network['idx_upstream_link'].to_numpy()
    _idxs = network['idx'].to_numpy()
    ncpu = 2
    manager = Manager()
    idx_upstream_link = manager.list(_idx_upstream_links)
    idx = manager.list(_idxs)
    _expr = [[] for x in range(4)]
    expr = manager.list(_expr)
    jobs = []
    for idx in _idxs[0:3]:
        order=1
        j = Process(target=process_unit,args=(idx,idx_upstream_link,order,expr[idx]))
        jobs.append(j)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    print('stop')

def test_process_all_multiprocessing():
    freeze_support()
    t = time.time()
    network = get_default_network()
    process_all_multiprocessing(network)
    print(time.time()-t)

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
    test_process_all_multiprocessing()
    # test_process_unit()
