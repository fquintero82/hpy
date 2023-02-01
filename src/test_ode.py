from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
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
    plt.show



