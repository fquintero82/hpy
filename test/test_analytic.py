import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from math import factorial


def solution_at_leafs(x:np.ndarray,
                       a:np.float32,
                       t:np.float32)->np.ndarray:
        '''
        solves diff equation dx/dt = -a*x(t) at network first order links
        x is a numpy array with discharge value at time t=0
        a is river velocity  / channel_length [m/s] / [m]
        output is discharge values at time t
        y(t) = c_2 e^(-a t)
        '''
        c = x
        out = c * np.exp(-a*t)
        return out

def solution_down(x):
        '''
        dx/dt =-a(x(t) + y(t) )
        x(t) = c_1 e^(-a t) - a c_2 t e^(-a t)
        '''
        pass


def test1():
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.integrate import solve_ivp
    
    x1 = 10
    x2 =1
    x3=0
    p = 1
    t = np.linspace(0,1)
    out1 = x1 * np.exp(-p*t)
    out2 = x2 * np.exp(-p*t) + p*x1*t*np.exp(-p*t)
    out3 = x3 * np.exp(-p*t) + np.power(p,1)*x2*np.power(t,1)*(1/factorial(1))*np.exp(-p*t)    + np.power(p,2)*x1*np.power(t,2)*(1/factorial(2))*np.exp(-p*t)
    def fun(t,x):
        p=1
        return [p*(-x[0]), p*(-x[1]+x[0]) ,p*(-x[2]+x[1])]
    
    x =[x1,x2,x3]
    res = solve_ivp(fun,t_span=(0,1),y0=x)

    plt.plot(t,out1,label='out1')
    plt.plot(t,out2,label='out2')
    plt.plot(t,out3,label='out3')
    plt.plot(res['t'],res['y'][0],label='1')
    plt.plot(res['t'],res['y'][1],label='2')
    plt.plot(res['t'],res['y'][2],label='3')

    plt.legend()
    plt.show()

test1()

