import os
import numpy as np

    
dir = '/Users/felipe/hio/hio/examples/HLM_Binaries'

def get_values(time: int,options=None):
    prefix = ''
    if options is not None:
        dir = options['path']
    if 'prefix' in list(options.keys()):
        prefix = options['prefix']
    filein = os.path.join(dir,prefix+str(time))
    lid = None
    val = 0
    try:
        with open(file=filein,mode='rb') as myfile:
            file  = open(file=filein,mode='rb')
            data = read_bin(file)
            lid = data['lid']
            val = data['val']
    except FileNotFoundError as e:
        print(e)
    return lid , val


def read_bin(path):
    return np.fromfile(path,
                    dtype=np.dtype([('lid', np.int32),('val', np.float32)]),
                    offset=4,) 



