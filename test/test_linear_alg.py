from test_dataframes import getTestDF2
import numpy as np
import pandas as pd
from numpy.linalg import inv
network = getTestDF2('network')
states = getTestDF2('states')
states['discharge'] = 1

import numpy  as np
from PyGMRES.gmres import GMRES
from PyGMRES.linop import laplace_2d_extrap, resid

#y0 = 1
#yt1 = y0 - out + in

# 5\
# 4-3-1
#   2/
def test1():
    #with f constant
    #equations
    q0 = np.ones(shape=(5,))
    x = np.zeros(shape=(5,))
    f = 1
    
    for t in range(4):
        print(q0)
        #future = past - output + input (from t0 only)
        #water drains to the next downstream link, but one drop of water in the
        #upstream channels do not make it to the basin outlet
        x[1-1] = q0[1-1] - f*q0[1-1] + f*q0[2-1] + f*q0[3-1]
        x[2-1] = q0[2-1] - f*q0[2-1] + 0
        x[3-1] = q0[3-1] - f*q0[3-1] + f*q0[4-1] + f*q0[5-1]
        x[4-1] = q0[4-1] - f*q0[4-1] + 0
        x[5-1] = q0[5-1] - f*q0[5-1]+ 0
        q0 = x

def test2():
    #vectorial form of test1
    network = pd.read_pickle('examples/small/small.pkl')
    q0 = np.ones(shape=(5,))
    f = 0.2
    A = np.array([
            [1-f,f,f,0,0],
            [0,1-f,0,0,0],
            [0,0,1-f,f,f],
            [0,0,0,1-f,0],
            [0,0,0,0,1-f],
            ])
    for i in range(20):
        x = np.matmul(A,q0)
        print(q0)
        q0=x

def test3():
    #f changes for each link
    q0 = np.ones(shape=(5,))
    x = np.zeros(shape=(5,))
    f = np.array([0.2, 0.3 , 0.1,0.1,0.2])
    
    for t in range(20):
        #future = past - output + input (from t0 only)
        x[1-1] = q0[1-1] - f[1-1]*q0[1-1] + f[2-1]*q0[2-1] + f[3-1]*q0[3-1]
        x[2-1] = q0[2-1] - f[2-1]*q0[2-1] + 0
        x[3-1] = q0[3-1] - f[3-1]*q0[3-1] + f[4-1]*q0[4-1] + f[5-1]*q0[5-1]
        x[4-1] = q0[4-1] - f[4-1]*q0[4-1] + 0
        x[5-1] = q0[5-1] - f[5-1]*q0[5-1]+ 0
        print(q0)
        q0 = x

def test4():
    #vectorial form of test3
    network = pd.read_pickle('examples/small/small.pkl')
    q0 = np.ones(shape=(5,))
    f = np.array([0.2, 0.3 , 0.1,0.1,0.2])
    A = np.array([
            [-1,1,1,0,0],
            [0,-1,0,0,0],
            [0,0,-1,1,1],
            [0,0,0,-1,0],
            [0,0,0,0,-1],
            ])
    for i in range(20):
        x = q0 + np.matmul(A,f*q0)
        print(q0)
        q0=x

def test5():
    #create adjacency matrix from network topology
    network = pd.read_pickle('examples/small/small.pkl')
    nlinks = len(network)
    A = np.eye(nlinks)*-1
    for ii in np.arange(nlinks)+1:
        print(ii)
        idx_up = network.at[ii,'idx_upstream_link']
        if np.array([idx_up !=-1]).any():
            A[ii-1,(idx_up-1).tolist()]=1
    print(A)

def test6():
    #accumulates along the drainage network. the result at each link is the total of the upstream channels
    # x: total of upstream channels
    # q: value at channel
    q0 = np.ones(shape=(5,))
    x = np.zeros(shape=(5,))
    #equations
    # future = present + acumulation upstream
    x[5-1] = q0[5-1]
    x[4-1] = q0[4-1]
    x[3-1] = q0[3-1] + x[4-1] + x[5-1]
    x[2-1] = q0[2-1]
    x[1-1] = q0[1-1] + x[2-1] + x[3-1]

    #rewriten
    # x[5-1] = q0[5-1]
    # x[4-1] = q0[4-1]
    # x[3-1] - x[4-1] - x[5-1]= q0[3-1] 
    # x[2-1] = q0[2-1]
    # x[1-1] - x[2-1] - x[3-1]= q0[1-1] 

    #AX=q0
    #X =inv(A)q0
    A = np.array([
            [1,-1,-1,0,0],
            [0,1,0,0,0],
            [0,0,1,-1,-1],
            [0,0,0,1,0],
            [0,0,0,0,1],
            ])
    x = np.matmul(inv(A),q0)
    print(x)
    network = pd.read_pickle('examples/small/small.pkl')
    nlinks = len(network)
    A = np.eye(nlinks)
    for ii in np.arange(nlinks)+1:
        print(ii)
        idx_up = network.at[ii,'idx_upstream_link']
        if np.array([idx_up !=-1]).any():
            A[ii-1,(idx_up-1).tolist()]=-1
    print(A)

def test7():

    # Define the linear system of equations
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([1, 2, 3])

    # Define the initial guess for the solution
    x0 = np.zeros_like(b)
    # Solve the linear system using GMRES
    tolerance = 1e-6
    max_iterations = 100
    solution, error = GMRES(A, b, x0, tolerance, max_iterations)

    # Print the solution
    print(x)
# def test7():
#     import numpy as np
#     from scipy import linalg
#     import time
#     n=1000 # 40000 killed my computer
#     A=np.random.rand(n,n)
#     start= time.time()
#     Am=np.linalg.inv(A.copy())
#     end= time.time()
#     print ('np.linalg.inv ', end-start, ' seconds')
#     print ('residual ', np.linalg.norm(A.dot(Am)-np.identity(n), np.inf))
