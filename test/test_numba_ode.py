import nbkode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from numbalsoda import lsoda,lsoda_sig
from numba import njit,cfunc
from numba import numba as nb
from utils.network.network import get_default_network
from hlm import HLM
#from utils.network.network_from_rvr_file import combine_rvr_prm
import time

def test1():
    def rhs(t, y):
        return -0.1 * y
    
    y0 = 1.
    t0 = 0
    # solver = nbkode.RungeKutta45(rhs, t0, y0)
    ts = [0,.5,1]
    start_time = time.time()
    solver = nbkode.ForwardEuler(rhs, t0, y0)

    # ts, ys = solver.run(ts)
    solver.skip(upto_t=1)
    print(solver.t)
    print(solver.y)
    print("--- %s seconds ---" % (time.time() - start_time))
    # plt.plot(ts, ys) 
    # plt.show()

def test2(t,q,p):
    def fun(t,q,p): #t in minutes, q in m3/h
        velocity = (0.1)
        channel_len_m = (500.0)
        #idx_up = p[2]
        #q_aux = np.zeros(shape=(q.shape[0]+1),dtype=np.float16)#fail
        q_aux = np.zeros(shape=(q.shape[0]+1)) #wont fail. dont pass type. numba infer it
        q_aux[1:] = q #dont fail
        #q_upstream = np.zeros(q.shape[0])
        #q_upstream = np.array([np.sum(q_aux[x]) for x in idx_up]) #m3/h
        #velocity *=60*60 #m/s to m/h
        #dq_dt = np.array((1/channel_len_m )* velocity * (-1*q_aux[1:] + q_upstream))
        dq_dt = ((1.0)/channel_len_m )* velocity #* (-1*q_aux[1:])
        return dq_dt

    rvr_file ='../examples/cedarrapids1/367813.rvr'
    prm_file ='../examples/cedarrapids1/367813.prm'
    network = combine_rvr_prm(prm_file,rvr_file)
    nlinks = network.shape[0]
    states = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(STATES_NAMES))),
        columns=STATES_NAMES)
    states['link_id'] = network['link_id'].to_numpy()
    states.index = states['link_id'].to_numpy()
    states['discharge']=1
    velocity = 0.5 #m/s
    idx_up = network['idx_upstream_link'].to_numpy()
    q = np.array(states['discharge'],dtype=np.float16) #force everything to float16
    t0=0
    channel_len_m = np.array(network['channel_length'])
    start_time = time.time()
    p =[velocity,channel_len_m,idx_up] #p must be array, not tuple?
    p=[0.1,0.2]
    solver = nbkode.RungeKutta45(fun, t0, q, params=p)
    ts, ys = solver.step(n=1)

def test3():
    #https://stackoverflow.com/questions/57706940/solving-ode-with-large-set-of-initial-conditions-in-parallel-through-python
    start_time = time.time()
    @cfunc(lsoda_sig)
    def rhs(t, u,du,p):
        du[0]=  u[0]-u[0]*u[1]
        du[1] = u[0]*u[1]-u[1]

    funcptr = rhs.address
    t_eval = np.linspace(0.0,20.0,201)
    np.random.seed(0)
    
    @nb.njit(parallel=True)
    def main(n):
        u1 = np.empty((n,len(t_eval)), np.float64)
        u2 = np.empty((n,len(t_eval)), np.float64)
        for i in nb.prange(n):
            u0 = np.empty((2,), np.float64)
            u0[0] = np.random.uniform(4.5,5.5)
            u0[1] = np.random.uniform(0.7,0.9)
            usol, success = lsoda(funcptr, u0, t_eval, rtol = 1e-8, atol = 1e-8)
            u1[i] = usol[:,0]
            u2[i] = usol[:,1]
        return u1, u2

    u1, u2 = main(10000)
    usol, success = lsoda(funcptr, u0, t_eval, data = data)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(usol)

def test4():
    @cfunc(lsoda_sig)
    def rhs(t, u,du,p):
        du[0]=  u[0]-u[0]*u[1]

def test5():
    hlm_object = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    # config_file = 'examples/hydrosheds/conus2.yaml'

    hlm_object.init_from_file(config_file)
    N = hlm_object.network.shape[0]
    channel_len_m = hlm_object.network['channel_length'].to_numpy()
    velocity = 3600*hlm_object.params['river_velocity'].to_numpy() #m/h
    y0 = np.ones(shape=(N))

    def rhs(t, y,p):
        return p * -y
    
    t = time.time()
    ts = [1]
    t0 = 0

    # solver = nbkode.RungeKutta45(rhs, t0, y0,params=channel_len_m/velocity)
    # solver = nbkode.ForwardEuler(rhs, t0, y0,params=channel_len_m/velocity)
    solver = nbkode.RungeKutta23(rhs, t0, y0,params=channel_len_m/velocity)

    
    ts, ys = solver.run(ts)

        # print(ys)
    print("--- %s seconds ---" % (time.time() - t))

test5()