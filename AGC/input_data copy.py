'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import logging

'''
    my func
'''

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    """
    load the following files:
    _ ind.cora.allx: <1708x1433 sparse matrix of type '<class 'numpy.float32'>'
        with 31261 stored elements in Compressed Sparse Row format>
    - ind.cora.ally: numpy.ndarray int32 of shape (1708, 7)  (one hot encoding)
    - ind.cora.graph: defaultdict(list)
    """    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    nx_graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(nx_graph)

    # y
    targets = np.concatenate((ally, ty))
    
    targets_id = np.argmax(targets, axis = 1)
    targets_id[test_idx_reorder] = targets_id[test_idx_range]   # x换序y也要换序
    classes = targets.shape[1]    

    logging.info('dataset massage:')
    logging.info('feature size: {}'.format(features.shape))

    return nx_graph, adj, features, targets_id, classes

'''
    origin func
'''
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx): 
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj_orig):
    adj = sp.coo_matrix(adj_orig)    # 没变啊。。
    # print(adj.toarray())
    adj_ = adj + sp.eye(adj.shape[0])   # 加自环
    # print(adj_.toarray())
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

