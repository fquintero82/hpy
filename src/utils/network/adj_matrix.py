def get_adjacency_matrix(network:pd.DataFrame,default=False):

    if default==False:
        nlinks = len(network)
        #A = np.eye(nlinks,dtype=np.byte)*-1
        A = np.eye(nlinks,dtype=np.float32)*-1
        for ii in np.arange(nlinks):
            idx_up = network.iloc[ii]['idx_upstream_link']
            if np.array([idx_up !=-1]).any():
                A[ii,(idx_up-1).tolist()]=1
        return A
    else:
        file1 = open('examples/cedarrapids1/367813_adj.pkl','rb')
        return pickle.load(file1)
        file1.close()