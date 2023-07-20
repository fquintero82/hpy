from hlm import HLM
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc

def test():
    instance = HLM()
    config_file = 'examples/hydrosheds/conus2.yaml'
    instance.init_from_file(config_file)
    instance.advance()   


if __name__ == "__main__":
    test()
    #plot1()

    
