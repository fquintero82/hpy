from hlm import HLM
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc
import cProfile

    
def test1():
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_macbook.yaml'
    instance.init_from_file(config_file,option_solver=False)
    for ii in range(5):
        print(ii)
        instance.advance_one_step()


def test4():
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_imac.yaml'
    instance.init_from_file(config_file,option_solver=False)
    instance.advance()

def test5():
    instance = HLM()
    config_file = 'examples/iowa/iowa_macbook.yaml'
    instance.init_from_file(config_file,option_solver=False)
    instance.advance()   

if __name__ == "__main__":
    test1()
    # cProfile.run('test1()')

    
