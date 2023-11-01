import time
import cupy as cp
import numpy as np

# create a random numpy array of size 1000x1000
a = np.random.rand(1000, 1000)

# create a random cupy array of size 1000x1000
b = cp.random.rand(1000, 1000)

# start timer
start = time.time()

# perform some random operations on numpy array
for i in range(1000):
    a = np.dot(a, a)

# print time taken
print("Time taken by numpy: ", time.time() - start)

# start timer
start = time.time()

# perform some random operations on cupy array
for i in range(1000):
    b = cp.dot(b, b)

# print time taken
print("Time taken by cupy: ", time.time() - start)