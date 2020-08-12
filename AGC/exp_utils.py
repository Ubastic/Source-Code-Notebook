import numpy as np
import scipy.sparse as sp
import networkx as nx
def exp_load_data(dataset, k):
    classes = 0
    if dataset=='dblp':
        classes = 4
    if dataset=='reut':
        classes = 4
    if dataset=='acm':
        classes = 3
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k)
    else:
        path = 'graph/{}_graph.txt'.format(dataset)
    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    #
    graph_list_tem = edges_unordered.tolist()
    temp = graph_list_tem
    graph = {}
    for i in range(len(graph_list_tem)):
        mylist = []
        for j in range(len(temp)):
            if graph_list_tem[i][0]==temp[j][0]:
                mylist.append(temp[j][1])
        graph[graph_list_tem[i][0]] = mylist

    nx_graph = nx.from_dict_of_lists(graph)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
    y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)
    return nx_graph, adj, x, y, classes

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

if __name__ == '__main__':
    exp_load_data('dblp',None)

