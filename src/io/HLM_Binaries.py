import os
import numpy as np

    
dir = '/Users/felipe/hio/hio/examples/HLM_Binaries'
def get_values(time: int,options=None):
    if options !=None:
        dir = options['path']
    filein = os.path.join(dir,str(time))
    with open(file=filein,mode='rb') as myfile:
        file  = open(file=filein,mode='rb')
        data = read_bin(file)
        return data['lid'] , data['val']


def read_bin(path):
    return np.fromfile(path,
                    dtype=np.dtype([('lid', np.int32),('val', np.float32)]),
                    offset=4,) 



