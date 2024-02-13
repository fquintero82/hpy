import os
import numpy as np

dir = '/Users/felipe/hio/hio/examples/HLM_Binaries'


def get_values(time: int, options=None):
    prefix = ''
    if options is not None:
        dir = options['path']  # look for a path key and assing its value to "dir"
    if 'prefix' in list(options.keys()):
        prefix = options['prefix']  # look for a "prefix" key
    filein = os.path.join(dir, prefix + str(time))
    lid = None
    val = 0
    try:
        with open(file=filein, mode='rb') as myfile: # remove redundant open file
            data = read_bin(myfile)
            lid = data['lid']
            val = data['val']

    # if the file does not exist, then lid is None, val is zero.
    except FileNotFoundError as e:
        pass
        print(e)

    return lid, val


def read_bin(path):
    return np.fromfile(path,
                       dtype=np.dtype([('lid', np.int32), ('val', np.float32)]),
                       offset=4, )




