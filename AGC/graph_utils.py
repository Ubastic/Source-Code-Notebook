import numpy as np
import pandas as pd
import networkx as nx

from scipy.sparse import linalg as la
from sklearn import metrics
from sklearn.cluster import SpectralClustering

import scipy.sparse as sp
import sys
import pickle as pkl
import logging


# empty
def construct_hypergraph_from_graph(adj_matrix):
    '''
    adj_matrix -> inc_matrix
    '''
    inc_matrix = adj_matrix
    
    return inc_matrix