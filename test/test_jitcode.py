from jitcode import jitcode,y
import numpy as np
from hlm import HLM
from matplotlib import pyplot as plt
import time

def test1():
    a  = -0.025794
    b1 =  0.0065
    b2 =  0.0135
    c  =  0.02
    k  =  0.128

    f = [
        y(0) * ( a-y(0) ) * ( y(0)-1.0 ) - y(1) + k * (y(2) - y(0)),
        b1*y(0) - c*y(1),
        y(2) * ( a-y(2) ) * ( y(2)-1.0 ) - y(3) + k * (y(0) - y(2)),
        b2*y(2) - c*y(3)
        ]

    initial_state = np.array([1.,2.,3.,4.])

    ODE = jitcode(f)
    ODE.set_integrator("dopri5")
    ODE.set_initial_value(initial_state,0.0)

    times = 2000+np.arange(100000)
    data = []
    for time in times:
        data.append(ODE.integrate(time))

    np.savetxt("test/timeseries.dat", data)

def test2():
    hlm_object = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    hlm_object.init_from_file(config_file)
    N = hlm_object.network.shape[0]
    channel_len_m = np.array(hlm_object.network['channel_length'])
    velocity = np.array(hlm_object.params['river_velocity']*3600) #m/h
    idx_up = hlm_object.network['idx_upstream_link'].to_numpy()
    #network indexes start at 1
    #index 0 is auxiliary to denote no  upstream
    #aux zero will be used for inputs
    #first element of f is zero
    f = [0]

    for i in np.arange(N)+1:
        coupling_sum = 0
        
        if (idx_up[i-1] !=0).any():
            print(i)
            print(idx_up[i-1])
            for j in idx_up[i-1]:
                coupling_sum = coupling_sum + y(j)
        #print(coupling_sum)
        f.append(
             (velocity[i-1] / channel_len_m [i-1]) * (-y(i)+coupling_sum)
        )
    
    initial_state = np.ones(shape=(N+1))
    initial_state[0] = 0
    ODE = jitcode(f)
    ODE.set_integrator("dopri5")
    ODE.set_initial_value(initial_state,0.0)
    times = np.arange(1,240)
    out = np.zeros(shape=(240))
    id = 32715
    
    for time in times:
        print(time)
        out[time] = ODE.integrate(time)[id]
    
    plt.plot(out)
    plt.show()

def test3():
    hlm_object = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    hlm_object.init_from_file(config_file)
    N = hlm_object.network.shape[0]
    idx_up = hlm_object.network['idx_upstream_link'].to_numpy()
    #network indexes start at 1
    #index 0 is auxiliary to denote no  upstream
    #aux zero will be used for inputs
    #first element of f is zero
    f = [0]
    for i in np.arange(N)+1:
        coupling_sum = 0
        if (idx_up[i-1] !=0).any():
            for j in idx_up[i-1]:
                coupling_sum = coupling_sum + y(j)
        f.append(coupling_sum)
    
    initial_state = np.ones(shape=(N+1))
    initial_state[0] = 0
    ODE = jitcode(f)
    ODE.set_integrator("RK45") #this line takes forever . test different methods
    ODE.set_initial_value(initial_state,0.0)
    times = 1
    out = ODE.integrate(times)
    out.max()
    from scipy import stats
    stats.describe(out)
    id = 32715
    out[id]
    plt.plot(out)
    plt.show()

def test4():
    # 5\
    # 4-3-1
    #   2/
    f = [0, #y0
         y(1)+y(3)+y(2), #y1
         y(2),
         y(3)+y(4),
         y(4),
         y(5)
         ]
    
    f = [0, #y0
         y(3)+y(2), #y1
         0, #2
         y(4)+y(5), #3
         0, #4
         0 #5
         ]
    
    init = [1,1,1,1,1,1]
    ODE = jitcode(f)
    ODE.set_integrator("dopri5") #this line takes forever . test different methods

    ODE.set_initial_value(init,0.0)
    times = 1
    out = ODE.integrate(times)
    print(out)

def test4():
    import js2py
    def sum_network(array,idxup,idxdown):
        funjs = """function sum_network(array,idxup,idxdown){
        const out = array.slice();
        for(let i=0; i< array.length; i++){
            if(idxdown[i] !==0){
                out[idxdown[i]-1]+=out[idxup[i]-1];
            }
        }
        return out;
        }"""
        
        sum_network_function = js2py.eval_js(funjs)
        return sum_network_function(array,idxup,idxdown)
    # 5\
    # 4-3-1
    #   2/
    array = [1,1,1,1,1]
    idxup = [2,4,5,3,1]
    idxdown = [1,3,3,1,0]
    sum_network(array,idxup,idxdown)

    hlm_object = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    hlm_object.init_from_file(config_file)
    N = hlm_object.network.shape[0]
    routing_order = hlm_object.network.loc[:,['idx','idx_downstream_link','drainage_area']].copy()
    routing_order = routing_order.sort_values(by=['drainage_area'])
    idxd = routing_order['idx_downstream_link'].to_numpy()
    idxu = routing_order['idx'].to_numpy()
    initial_state = np.ones(shape=(N+1))
    initial_state[0] = 0
    t = time.time()
    out = sum_network(initial_state,idxu,idxd)
    print(time.time()-t)
    
