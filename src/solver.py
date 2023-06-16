from jitcode import jitcode,y
import numpy as np

def create_solver(hlm_object):
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
            #print(i)
            #print(idx_up[i-1])
            for j in idx_up[i-1]:
                coupling_sum = coupling_sum + y(j)
        #print(coupling_sum)
        f.append(
             (velocity[i-1] / channel_len_m [i-1]) * (-y(i)+coupling_sum)
        )
    
    ODE = jitcode(f)
    ODE.set_integrator("dopri5")
    return ODE
