import pandas as pd
import numpy as np
import sys

def _get_basin_ids(idx:np.int32,
                    idx_upstream_links:np.ndarray,
                    expr:list):
    expr.append(idx)
    myidx_upstream_links = idx_upstream_links[idx - 1]
    if (myidx_upstream_links!=0).any():
        for new_idx in myidx_upstream_links:
            _get_basin_ids(new_idx,idx_upstream_links,expr)

def get_basin_ids(idx:np.int32,idx_upstream_links:np.ndarray):
    expr = []
    _get_basin_ids(idx,idx_upstream_links,expr)
    return expr

def network_subset(network:pd.DataFrame,outlet:int):
    idx_upstream_links = network['idx_upstream_link'].to_numpy()
    wh = np.where(network['link_id'].to_numpy()==outlet)[0][0]
    idxs = network['idx'].to_numpy()
    id = idxs[wh]
    sys.setrecursionlimit(int(1E6))
    x = get_basin_ids(id,idx_upstream_links)
    sys.setrecursionlimit(int(1E3))
    basin_idx = np.array(x[1:len(x):2])
    mask = np.isin(idxs, basin_idx)
    subbasin = network[mask]
    return subbasin

f = '/Users/felipe/tmp/conus_imac.pkl'
df = pd.read_pickle(f)
outlet = 70814357
subbasin = network_subset(df,outlet)
f = '/Users/felipe/tmp/ms_imac.pkl'
subbasin.to_pickle(f)