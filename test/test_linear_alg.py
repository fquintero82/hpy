from test_dataframes import getTestDF2
import numpy as np
network = getTestDF2('network')
states = getTestDF2('states')
states['discharge'] = 1

#y0 = 1
#yt1 = y0 - out + in

# 5\
# 4-2-1
#   3/

#equations
q0 = np.ones(shape=(5,))
y = np.zeros(shape=(5,))
f = 0.5
#future = past - output + input (from t0 only)
y[5-1] = q0[5-1] - f*q0[5-1]+ 0
y[4-1] = q0[4-1] - f*q0[4-1] + 0
y[3-1] = q0[3-1] - f*q0[3-1] + 0
y[2-1] = q0[2-1] - f*q0[2-1] + f*q0[4-1] + f*q0[5-1]
y[1-1] = q0[1-1] - f*q0[1-1] + f*q0[2-1] + f*q0[3-1]

input_upstream = np.array([0,0,0,f*q0[4-1] + f*q0[5-1],f*q0[2-1] + f*q0[3-1]])
q0 = y

