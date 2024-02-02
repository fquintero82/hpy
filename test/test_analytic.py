import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from  scipy.special import factorial
from hlm import HLM
import pandas as pd
from utils.network.network import get_default_network, update_network_pickle,get_network_from_file
import numba
from multiprocessing import Pool, Manager, Process,freeze_support
import time
import pickle
import warnings
from numba import njit,prange
from utils.network.network_symbolic import process_all,process_unit,eval_all,_get_routing_term_indices,_pack_to_df,_eval5,_eval4
from models.routing import transfer9, transfer10

def eval_unit(x:list,P:np.ndarray,X:np.ndarray,T:int):
    #uneven values of x are the order
    #even values of x are the index
    n=len(P)
    order = np.array(x[0:len(x):2])
    idx = np.array(x[1:len(x):2])
    val = 1/(factorial(order-1)) * P[idx-1]**(order-1) * X[idx-1] * T**(order-1) * 2.718 ** (-P[idx-1] * T)
    out = np.sum(val)
    return out

def test1():
    X = np.array([10,1,0])
    P = np.ones(3)*0.1
    T = np.linspace(0,10)
    # T = 1
    x =[1,1]
    out1 = [eval_unit(x,P,X,t) for t in T]
    x = [1,2,2,1]
    out2 = [eval_unit(x,P,X,t) for t in T]
    x = [1,3,2,2,3,1]
    out3 = [eval_unit(x,P,X,t) for t in T]
    
    def fun(t,X):
        p=0.1
        return [p*(-X[0]), p*(-X[1]+X[0]) ,p*(-X[2]+X[1])]
    
   
    res = solve_ivp(fun,t_span=(0,10),y0=X)

    plt.plot(T,out1,label='out1')
    plt.plot(T,out2,label='out2')
    plt.plot(T,out3,label='out3')
    plt.plot(res['t'],res['y'][0],label='1')
    plt.plot(res['t'],res['y'][1],label='2')
    plt.plot(res['t'],res['y'][2],label='3')

    plt.legend()
    plt.show()

def test1b():
    X = np.array([1,1,1,1,1])
    P = np.ones(5)*1
    T = np.linspace(0,10)
    # T = 1
    x =[1,1,2,2,2,3,3,4,3,5]
    out1 = [eval_unit(x,P,X,t) for t in T]
    x = [1,2]
    out2 = [eval_unit(x,P,X,t) for t in T]
    x = [1,3,2,4,2,5]
    out3 = [eval_unit(x,P,X,t) for t in T]
    x = [1,4]
    out4 = [eval_unit(x,P,X,t) for t in T]
    x = [1,5]
    out5 = [eval_unit(x,P,X,t) for t in T]

    def fun(t,X):
        p=1
        # return [p*(-X[0]), p*(-X[1]+X[0]) ,p*(-X[2]+X[1])]
        return [p*(-X[0]+X[1]+X[2]), #X[0]
                p*(-X[1]),  #X[1]
                p*(-X[2]+X[3]+X[4]), #X[2]
                p*(-X[3]),
                p*(-X[4])]
   
    res = solve_ivp(fun,t_span=(0,10),y0=X)

    instance = HLM()
    config_file = 'examples/small/small.yaml'
    instance.init_from_file(config_file)
    for ii in range(10):
        transfer10(instance)

    df = pd.DataFrame({'tan':T,
                       'a1':np.array(out1),
                       'a2':np.array(out2),
                       'a3':np.array(out3),
                       'a4':np.array(out4),
                       'a5':np.array(out5),
                       })
    df.to_csv('/Users/felipe/tmp/analytic_numeric/analytic.csv',index=False)
    df = pd.DataFrame({'tnum':np.array(res['t']),
                       'n1':np.array(res['y'][0]),
                       'n2':np.array(res['y'][1]),
                       'n3':np.array(res['y'][2]),
                       'n4':np.array(res['y'][3]),
                       'n5':np.array(res['y'][4]),
                       })
    df.to_csv('/Users/felipe/tmp/analytic_numeric/numeric.csv',index=False)

    plt.plot(T,out1,label='out1')
    plt.plot(T,out2,label='out2')
    plt.plot(T,out3,label='out3')
    plt.plot(T,out4,label='out4')
    plt.plot(T,out5,label='out5')
    plt.plot(res['t'],res['y'][0],label='1')
    plt.plot(res['t'],res['y'][1],label='2')
    plt.plot(res['t'],res['y'][2],label='3')
    plt.plot(res['t'],res['y'][3],label='4')
    plt.plot(res['t'],res['y'][4],label='5')
    plt.legend()
    plt.show()

def test1c():
    id = 558284
    ad=0.31067
    l= 0.88*1000#km->m
    ai=0.31067*1e6#m2
    real =[1.00,0.51,0.26,0.13,0.07,0.03,0.02,0.01,0.00]
    v = 0.75#m/s
    invtau = 60.0*v*ai**0 / ((1.0 - 0)*l)	#[1/min]  invtau
    #dqdt = invtau * pow(q, 0) * ans[0];
    
    import numpy as np
    def fun(t,X):
        l= 0.88*1000#km->m        
        v = 0.75#m/s
        invtau = 60.0*v / l	#[1/min]  invtau
        invtau #0.051
        # invtau = 0.011
        return (invtau*(-X))
    # Runge-Kutta method, dx/dt = -1*0.75/(0.88*1000)x, x(0) = 1, from 0 to 7200, h = 60
    X=[1] #m3/s 
    X = [60] #m3/min
    res = solve_ivp(fun,t_span=[0,600],t_eval=[0,60,120,180],y0=X)
    #res = solve_ivp(fun,t_span=[0,600],t_eval=[0,59,119,179],y0=X)
    np.set_printoptions(suppress=True)
    y = np.array(res['y'][0])
    print(y/60.)

def test_eval_unit():
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    instance.init_from_file(config_file,option_solver=False)
    idx = 32715

    N=len(instance.network)
    expr = instance.network['expression'].to_numpy()
    P = (instance.params['river_velocity'] / instance.network['channel_length']).to_numpy()
    T=instance.time_step_sec
    X = np.ones(shape=(N,))
    val = eval_unit(expr[idx],P,X,T)

def test_process_unit():
    network = get_default_network()
    N = len(network)
    link_id = 367813
    idx = 32715
    idx = 40163
    idx_upstream_links = network['idx_upstream_link'].to_numpy()
    out = process_unit(idx,idx_upstream_links)
    print('ok')

# def process_all_map(network:pd.DataFrame):
#     N = len(network)
#     idx_upstream_links = network['idx_upstream_link'].to_numpy()
#     idxs = network['idx'].to_numpy()
#     out = list(map(process_unit,idxs[0:9],idx_upstream_links))

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

class Evaluator(object):
    def __init__(self, dict):
        self.dict = dict
    def __call__(self, src):
        eval(src, self.dict)

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
    with Pool(processes=8) as pool:
        out = pool.map(Evaluator(d),expr)
    print('done in %f sec'%(time.time()-t))
    
def test_process_all():
    f = 'examples/cedarrapids1/367813.pkl'
    network = pd.read_pickle(f)
    process_all(network)


def test_eval_all():
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    instance.init_from_file(config_file,option_solver=False)
    N=len(instance.network)
    process_all(instance.network)
    expr = instance.network['expression'].to_numpy()
    P = (instance.params['river_velocity'] / instance.network['channel_length']).to_numpy()
    T=instance.time_step_sec
    X = np.ones(shape=(N,))
    eval_all(expr,P,X,T)

def test_polars_eval():
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_imac.yaml'
    instance.init_from_file(config_file,option_solver=False)
    routing_term, idx,source_idx = _get_routing_term_indices(instance)
    t=time.time()
    routing_df = _pack_to_df(routing_term, idx,source_idx)
    X = np.ones(len(instance.network))
    df2 = _eval5(idx,X,routing_df)
    x = int(1000*(time.time()-t))
    print('done in {x} msec'.format(x=x))

def test_numpy_eval():
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_imac.yaml'
    instance.init_from_file(config_file,option_solver=False)
    routing_term, idx,source_idx = _get_routing_term_indices(instance)
    X = np.ones(len(instance.network))
    t=time.time()
    out = _eval4(idx,X,routing_term,source_idx)
    x = int(1000*(time.time()-t))
    print('done in {x} msec'.format(x=x))
# def process_all_map(network:pd.DataFrame):
#     N = len(network)
#     idx_upstream_links = network['idx_upstream_link'].to_numpy()
#     idxs = network['idx'].to_numpy()
#     expression=np.chararray(shape=(N,))

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
    test1b()
    # test_polars_eval()
    # test_numpy_eval()
    # test_process_all()
    # test_eval_unit()
    # process_all()
    # test_eval()
    # test_process_all_multiprocessing()
    # test_process_unit()

