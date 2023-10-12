from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import numpy as np
from test_dataframes import getTestDF1, getTestDF2
import pandas as pd
from utils.network.network import combine_rvr_prm
from model400names import STATES_NAMES
import time
# import tensorflow as tf
# import tensorflow_probability as tfp
from utils.network.network import get_default_network
from hlm import HLM
# from PyDSTool import *

# http://www.stochasticlifestyle.com/comparison-differential-equation-solver-suites-matlab-r-julia-python-c-fortran/
def test1():
    '''
    #v[0] = V11'(s) = -12*v12(s)**2 
    #v[1] v22'(s) = 12*v12(s)**2  
    #v[2] v12'(s) = 6*v11(s)*v12(s) - 6*v12(s)*v22(s) - 36*v12(s) 
    '''
    def fun(s,v):
        return [-12*v[2]**2 , 12*v[2]**2, 6*v[0]*v[2]-6*v[2]*v[1] - 36*v[2]]

    res = solve_ivp(fun,
        t_span=(0,0.1),
        y0=[2,3,4]
    )
    res['y'].shape
    plt.plot(res['t'],res['y'][0])
    plt.plot(res['t'],res['y'][1])
    plt.plot(res['t'],res['y'][2])
    plt.show()

def test2():
    #channel 1 and 2 drain to 3
    def fun(t,q):
        v0=0.05 #[m/s]
        lambda1= 0.0#0.33
        lambda2= 0#0.1
        A_i=1     #[km2]
        L_i = 100   #[m]
        invtau = (v0*A_i**lambda2) /((1.0 - lambda1)*L_i)
        return [invtau*(q[0]**lambda1)*(-1*q[0]),
                invtau*(q[1]**lambda1)*(-1*q[1]),
                invtau*(q[2]**lambda1)*(-1*q[2]+q[0]+q[1])
        ]
    q =[100,110,100]
    DT = 60 #min
    t_end = DT * 60 #seg
    res = solve_ivp(fun,t_span=(0,t_end),y0=q)
    plt.plot(res['t'],res['y'][0])
    plt.plot(res['t'],res['y'][1])
    plt.plot(res['t'],res['y'][2])
    plt.show()

def test3():
    #channel 1 and 2 drain to 3
    def fun(t,q,v0,lambda1,lambda2,A_i,L_i):
        invtau = (v0*A_i**lambda2) /((1.0 - lambda1)*L_i)
        return [invtau*(q[0]**lambda1)*(-1*q[0]),
                invtau*(q[1]**lambda1)*(-1*q[1]),
                invtau*(q[2]**lambda1)*(-1*q[2]+q[0]+q[1])
        ]
    args=(.3,.33,.1,100,200)
    res = solve_ivp(fun,t_span=(0,0.1),y0=[100,110,100],args=args)


def test3():
    #channel 1 and 2 drain to 3
    def fun(t,q,v0,invtau,index_up1,index_up2):
        #invtau = (v0*A_i**lambda2) /((1.0 - lambda1)*L_i)
        #invtau = np.divide(np.multiply(v0,np.power(A_i,lambda2)),np.multiply(np.subtract(1,lambda1),L_i))
        #print(q.shape)
        return np.multiply(np.power(np.multiply(invtau,q),lambda1),(-1*q+q[index_up1]+q[index_up2]))


    v0=np.array([0, .3, .4 , .5])
    lambda1=np.array([0, 0.33,.33,.33])
    lambda2=np.array([0, .1,.1,.1])
    A_i=np.array([0, 100,100,100])    #km2
    L_i = np.array([0, 100,100,100])   #m
    invtau = np.divide(np.multiply(v0,np.power(A_i,lambda2)),np.multiply(np.subtract(1,lambda1),L_i)) 
    args1 = (v0,lambda1,lambda2,A_i,L_i)
    args2=(.3,.33,.1,100,200)
    index_up1 = [0,0,0,1]
    index_up2 = [0,0,0,2]
    res = solve_ivp(fun,t_span=(0,0.1),y0=[0,100,110,100],args=args1,)
    res = solve_ivp(fun,t_span=(0,0.1),y0=[0,100,110,100],args=args2,)
    plt.plot(res['t'],res['y'][0])
    plt.plot(res['t'],res['y'][1])
    plt.plot(res['t'],res['y'][2])
    plt.show()

def test4():
    #[-3.243743104813389, -3.6821266577313194, 3.568117415294728]
    #0.007096536682661702
    states= getTestDF1('states')
    states['discharge'] = [100,110,100]
    q = states['discharge']
    q_aux = pd.concat([
            pd.Series(0,index=[0]),
            pd.Series(states['discharge'])
        ])
    params= getTestDF1('params')
    network= getTestDF1('network')
    #q_upstream = [np.sum(q_aux[x]) for x in network['upstream_link']]
    invtau = np.divide(
                np.multiply(
                    params['river_velocity'],
                    np.power(params['drainage_area'],params['lambda2'])
                ),
                np.multiply(
                    np.subtract(1,params['lambda1'])
                    ,params['channel_length']
                )
    )
    
    def fun(t,q,invtau,idx_up,lambda1):
        q_aux = pd.concat([
             pd.Series(0,index=[0]),
             pd.Series(q)
        ]).to_numpy() #it is important to convert this pd df into a nparray,otherwise i got broadcast errors
        q_upstream = np.array([np.sum(q_aux[x]) for x in idx_up])
        #q_aux = q_aux[1:]
        #dq_dt = invtau*np.power(q_aux.iloc[1:],lambda1)*(-1*q_aux.iloc[1:] + q_upstream)
        #dq_dt = invtau*q_aux.iloc[1:]**lambda1*(-1*q_aux.iloc[1:] + q_upstream)
        dq_dt = invtau*q_aux[1:]**lambda1*(-1*q_aux[1:] + q_upstream)
        #dq_dt = invtau*np.power(pd.concat([pd.Series(0,index=[0]),pd.Series(q)]),lambda1)*(-1*pd.concat([pd.Series(0,index=[0]),pd.Series(q)]) + [np.sum(pd.concat([pd.Series(0,index=[0]),pd.Series(q)])[x]) for x in idx_up])
        return dq_dt

    idx_up = network['upstream_link']
    lambda1 =params['lambda1']
    res = solve_ivp(fun,t_span=(0,1000),y0=q,args=(invtau,idx_up,lambda1))
    plt.plot(res['t'],res['y'][0])
    plt.plot(res['t'],res['y'][1])
    plt.plot(res['t'],res['y'][2])
    plt.show()

def test5():
    states= getTestDF2('states')
    states['discharge'] = [100,110,120,130,140]
    q = states['discharge']
    q_aux = pd.concat([
            pd.Series(0,index=[0]),
            pd.Series(states['discharge'])
        ])
    params= getTestDF2('params')
    network= getTestDF2('network')
    #q_upstream = [np.sum(q_aux[x]) for x in network['upstream_link']]
    invtau = np.divide(
                np.multiply(
                    params['river_velocity'],
                    np.power(params['drainage_area'],params['lambda2'])
                ),
                np.multiply(
                    np.subtract(1,params['lambda1'])
                    ,params['channel_length']
                )
    )
    
    def fun(t,q,invtau,idx_up,lambda1):
        q_aux = pd.concat([
             pd.Series(0,index=[0]),
             pd.Series(q)
        ]).to_numpy() #it is important to convert this pd df into a nparray,otherwise i got broadcast errors
        q_upstream = np.array([np.sum(q_aux[x]) for x in idx_up])
        dq_dt = invtau*q_aux[1:]**lambda1*(-1*q_aux[1:] + q_upstream)
        return dq_dt

    idx_up = network['upstream_link']
    lambda1 =params['lambda1']
    DT = 60
    t_end = DT*60 #secs
    res = solve_ivp(fun,t_span=(0,t_end),y0=q,args=(invtau,idx_up,lambda1))
    n_eval = res.t.shape[0] 
    y_1 = res.y[:,n_eval-1]
    for x in range(5):
        plt.plot(res['t'],res['y'][x],label=str(x+1))
    plt.legend()
    plt.show()

def test6():
    def fun(t,q,velocity,channel_len_m,idx_up): #t in minutes, q in m3/h
        #print(type(q))
        q_aux = np.concatenate(([0],q),dtype=np.float32)
        #q_upstream = np.zeros(q.shape)
        q_upstream = np.array([np.sum(q_aux[x]) for x in idx_up],dtype=np.float32) #m3/h
        velocity *=60*60 #m/s to m/h
        dq_dt = (1/channel_len_m )* velocity * (-1*q_aux[1:] + q_upstream)
        return dq_dt

    #rvr_file ='../examples/cedarrapids1/367813.rvr'
    #prm_file ='../examples/cedarrapids1/367813.prm'
    #network = combine_rvr_prm(prm_file,rvr_file)
    network = get_default_network()
    nlinks = network.shape[0]
    states = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(STATES_NAMES))),
        columns=STATES_NAMES)
    states['link_id'] = network['link_id'].to_numpy()
    states.index = states['link_id'].to_numpy()
    states['discharge']=1
    velocity = 0.5 #m/s
    idx_up = network['idx_upstream_link'].to_numpy()
    q = np.array(states['discharge'],dtype=np.float32)
    channel_len_m = np.array(network['channel_length'])
    start_time = time.time()
    res = solve_ivp(fun,
            t_span=(0,1),
            y0=q*60*60,
            args=(velocity,channel_len_m,idx_up),
            vectorized=False
        )
    print("--- %s seconds ---" % (time.time() - start_time))
    wh = np.where(network['link_id']==367813)[0][0]
    qout = res['y'][wh]/3600. #m3/h to m3/s
    plt.plot(res['t'],qout,label='outlet')
    plt.legend()
    plt.show()

#https://github.com/Nicholaswogan/NumbaLSODA

def test7():
    #test vectorization
    def fun(t,q,velocity,channel_len_m,idx_up): #t in minutes, q in m3/h
        #print(q.shape)
        #q=q.reshape(q.shape[0],1)
        _a = np.array([0])
        _a = _a.reshape(1,1)
        q_aux = np.concatenate((_a,q))
        #q_aux= q_aux.reshape(q.shape[0]+1,1)
        q_upstream = np.array([np.sum(q_aux[x]) for x in idx_up]) #m3/h
        q_upstream = q_upstream.reshape(q.shape[0],1)
        velocity *=60*60 #m/s to m/h
        channel_len_m = channel_len_m.reshape(channel_len_m.shape[0],1)
        dq_dt = (1/channel_len_m )* velocity * (-1*q_aux[1:] + q_upstream)
        #using np operators does not improve performance
        #dq_dt= np.multiply(np.divide(1,channel_len_m) ,velocity)
        #dq_dt= np.multiply(dq_dt,np.add(np.multiply(-1,q),q_upstream))
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
    velocity = 0.1 #m/s
    idx_up = network['idx_upstream_link'].to_numpy()
    q = np.array(states['discharge'])
    channel_len_m = np.array(network['channel_length'])
    start_time = time.time()
    res = solve_ivp(fun,
            t_span=(0,1),
            y0=q*60*60,
            args=(velocity,channel_len_m,idx_up),
            vectorized=True
        )
    print("--- %s seconds ---" % (time.time() - start_time))
    wh = np.where(network['link_id']==367813)[0][0]
    qout = res['y'][wh]/3600. #m3/h to m3/s
    plt.plot(res['t'],qout,label='outlet')
    plt.legend()
    plt.show()

def test8():
    hlm_object = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    hlm_object.init_from_file(config_file)

    def fun(t,v,velocity,channel_len_m,idx_up): #t in minutes, q in m3/h
        #print(type(q))
        v_aux = np.concatenate(([0],v),dtype=np.float64)
        #v_upstream = np.zeros(v.shape,dtype=np.float32)
        v_upstream = np.array([np.sum([v_aux[x]]) for x in idx_up])
        velocity = np.multiply(velocity,3600) #m/s to m/h
        dv_dt = (1/channel_len_m )* velocity * (-1*v_aux[1:] + v_upstream)
        #dv_dt = 0.5 * (-1*v_aux[1:] + v_upstream)
        return dv_dt

    idx_up = hlm_object.network['idx_upstream_link'].to_numpy()
    #t_end_sim = DT*60 #since the ODE flow inputs are in m3/s , t_end_sim is the number of seconds of routing process
    #t_end_sim = hlm_object.time_step_sec / 3600 # in hours
    v = np.array(hlm_object.states['volume'],dtype=np.float32)
    v = np.ones(shape=(hlm_object.network.shape[0]))

    channel_len_m = np.array(hlm_object.network['channel_length'],dtype=np.float64)
    velocity = np.array(hlm_object.params['river_velocity'],dtype=np.float64)
    start_time = time.time()
    res = solve_ivp(fun,
            t_span=(0,1),
            y0=v, 
            args=(velocity,channel_len_m,idx_up)
            ,method='RK23'
            ,atol=1e-2
            ,rtol=1e-2
            # ,vectorized=False
        )
    print("--- %s seconds ---" % (time.time() - start_time))
    n_eval = res.t.shape[0] 
    y_1 = res.y[:,n_eval-1]
    hlm_object.states['volume'] = np.array(y_1,dtype=np.float32)
    wh = np.where(hlm_object.network['link_id']==444657)[0][0] #444657 mason scity
    vout = res['y'][wh]
    plt.plot(res['t'],vout,label='outlet')
    plt.legend()
    plt.show()

def test9():
    hlm_object = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    hlm_object.init_from_file(config_file)

    def fun(t,v,velocity,channel_len_m,idx_up): #t in minutes, q in m3/h
        #print(type(q))
        zero = np.array([[0]])
        v_aux = np.concatenate((zero,v.reshape(-1,1)),dtype=np.float32)
        v_upstream = np.array([np.sum([v_aux[x]]) for x in idx_up]).reshape(-1,1)
        
        velocity = np.multiply(velocity,3600) #m/s to m/h
        
        dv_dt = (1/channel_len_m )* velocity * (-1*v_aux[1:] + v_upstream)
        #dv_dt = 0.5 * (-1*v_aux[1:] + v_upstream)
        return dv_dt

    idx_up = hlm_object.network['idx_upstream_link'].to_numpy()
    #t_end_sim = DT*60 #since the ODE flow inputs are in m3/s , t_end_sim is the number of seconds of routing process
    #t_end_sim = hlm_object.time_step_sec / 3600 # in hours
    v = np.array(hlm_object.states['volume'],dtype=np.float32)
    channel_len_m = np.array(hlm_object.network['channel_length'],dtype=np.float32).reshape(-1,1)
    velocity = np.array(hlm_object.params['river_velocity'],dtype=np.float32).reshape(-1,1)
    start_time = time.time()
    res = solve_ivp(fun,
            t_span=(0,1),
            y0=v, 
            args=(velocity,channel_len_m,idx_up),
            method='RK23',
            atol=1e-2,
            rtol=1e-2,
            vectorized=False
        )
    print("--- %s seconds ---" % (time.time() - start_time))
    n_eval = res.t.shape[0] 
    y_1 = res.y[:,n_eval-1]
    hlm_object.states['volume'] = np.array(y_1,dtype=np.float32)
    wh = np.where(hlm_object.network['link_id']==444657)[0][0] #444657 mason scity
    vout = res['y'][wh]
    plt.plot(res['t'],vout,label='outlet')
    plt.legend()
    plt.show()

def test10():
    icdict = {'x': 1, 'y': 0.4}    # Initial conditions dictonnary
    pardict = {'k': 0.1, 'm': 0.5} # Parameters values dictionnary
    x_rhs = 'y'
    y_rhs = '-k*x/m'
    vardict = {'x': x_rhs, 'y': y_rhs}
    DSargs = args()                   # create an empty object instance of the args class, call it DSargs
    DSargs.name = 'SHM'               # name our model
    DSargs.ics = icdict               # assign the icdict to the ics attribute
    DSargs.pars = pardict             # assign the pardict to the pars attribute
    DSargs.tdata = [0, 20]            # declare how long we expect to integrate for
    DSargs.varspecs = vardict         # assign the vardict dictionary to the 'varspecs' attribute of DSargs
    DS = Generator.Vode_ODEsystem(DSargs)
    DS.set(pars={'k': 0.3},
           ics={'x': 0.4})
    traj = DS.compute('demo')
    pts = traj.sample()
    plt.plot(pts['t'], pts['x'], label='x')
    plt.plot(pts['t'], pts['y'], label='y')
    plt.legend()
    plt.xlabel('t')
    plt.show()


def test_odeint1():
    def fun(q,t,velocity,channel_len_m,idx_up): #t in minutes, q in m3/h
        #print(type(q))
        q_aux = np.concatenate(([0],q))
        q_upstream = np.zeros(q.shape)
        q_upstream = np.array([np.sum(q_aux[x]) for x in idx_up]) #m3/h
        velocity *=60*60 #m/s to m/h
        dq_dt = (1/channel_len_m )* velocity * (-1*q_aux[1:] + q_upstream)
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
    q = np.array(states['discharge'])
    channel_len_m = np.array(network['channel_length'])
    start_time = time.time()
    res = solve_ivp(fun,
            t_span=(0,1),
            y0=q*60*60,
            args=(velocity,channel_len_m,idx_up),
            vectorized=False
        )
    print("--- %s seconds ---" % (time.time() - start_time))
    wh = np.where(network['link_id']==367813)[0][0]
    qout = res['y'][wh]/3600. #m3/h to m3/s
    plt.plot(res['t'],qout,label='outlet')
    plt.legend()
    plt.show()

def test_tf1():
    #https://computationalmindset.com/en/neural-networks/ordinary-differential-equation-solvers.html
    t_init, t0, t1 = 0., 0.5, 1.
    y_init = tf.constant([1., 1.], dtype=tf.float64)
    A = tf.constant([[-1., -2.], [-3., -4.]], dtype=tf.float64)

    def ode_fn(t, y):
        return tf.linalg.matvec(A, y)

    results = tfp.math.ode.BDF().solve(ode_fn, t_init, y_init,
                                   solution_times=[t0, t1])
    y0 = results.states[0]  # == dot(matrix_exp(A * t0), y_init)
    y1 = results.states[1]  # == dot(matrix_exp(A * t1), y_init)



def test11():
    hlm_object = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    hlm_object.init_from_file(config_file)
    N = hlm_object.network.shape[0]
    channel_len_m = hlm_object.network['channel_length'].to_numpy()
    velocity = 3600*hlm_object.params['river_velocity'].to_numpy() #m/h
    q = [1]

    def fun(t,q,velocity,channel_len_m): #t in minutes, q in m3/h
        velocity *=60*60 #m/s to m/h
        dq_dt = (1/channel_len_m )* velocity * (-q)
        return dq_dt
    
    t = time.time()
    for i in range(10):
        print(i)
        res = solve_ivp(fun,
            t_span=(0,1),
            y0=q*3600,
            args=(velocity[i],channel_len_m[i])
            )
        print(res)
    print("--- %s seconds ---" % (time.time() - t))

test11()     
    