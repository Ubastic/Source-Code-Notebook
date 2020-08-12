'''
todo:
1. 类间距离计算问题
2. 精度太低
3. 数据集增加
'''
import warnings

warnings.filterwarnings('ignore')
from parsertt import parameter_parser,tab_printer
from agc import agc
from utils import misc
from data_op import load_data
from exp_utils import *
from sklearn.metrics import adjusted_rand_score as ari_score
import time
import numpy as np
import pandas as pd
import networkx as nx

from sklearn import metrics
import scipy.sparse as sp
import sys
# import pickle as pkl
import logging
self_loop = False
def main():
    """
    """
    args = parameter_parser()
    tab_printer(args)

    misc.create_directory('./_out/')
    misc.setup_logger('agc', args, './_out/')
    logger = logging.getLogger()
    logging.getLogger().setLevel(logging.INFO)

    if args.dataset in ['citeseer', 'cora', 'pubmed']:
        nx_graph, adj, features, targets_id, classes = load_data(args.dataset)
    elif args.dataset in ['dblp', 'reut', 'acm']:
        if args.dataset == 'reut':
            nx_graph, adj, features, targets_id, classes = exp_load_data(args.dataset,3)
        else:
            nx_graph, adj, features, targets_id, classes = exp_load_data(args.dataset,None)
    elif args.dataset in ['wiki']:
        nx_graph = nx.from_edgelist(pd.read_csv('./data/wiki.cites.csv').values.tolist())
        features_file = pd.read_csv('./data/wiki.content.csv')
        data = np.array(features_file["content"].values.tolist())
        x1 = np.array(features_file["x1"].values.tolist())
        x2 = np.array(features_file["x2"].values.tolist())

        features = sp.csc_matrix((data, (x1, x2)))
        logging.info('dataset massage:')
        logging.info('feature size: {}'.format(features.shape))

        # nodes, targets_id = misc.target_reader('./data/wiki.label.csv')
        tar_file = pd.read_csv('./data/wiki.label.csv')
        nodes = tar_file["node"].values.tolist()
        targets_id = np.array(tar_file["labelId"]).reshape(-1,1).T[0]
        classes = 17

        adj = sp.csr_matrix(nx.adjacency_matrix(nx_graph))
    else:
        raise Exception("dataset import error.")
    # logging.info(features)     # 三元组形式
    # logging.info(targets)
    # logging.info(node)
    # logging.info(graph.adj)      # 邻接表
    # logging.info(nx.adjacency_matrix(graph))  # 临阶矩阵    ( .todense()转矩阵形式 )
    # logging.info(nx.degree(graph))   # 每个点的度

    start_time = time.time()
    logging.info("Timing begin")

    # agc
    if args.dataset in ['citeseer', 'cora', 'pubmed','wiki']:
        predict_C, epoch = agc(nx_graph, adj, features, targets_id, classes, start_time,1)
    if args.dataset in ['dblp', 'reut', 'acm']:
        predict_C, epoch = agc(nx_graph, adj, features, targets_id, classes, start_time,0)

    # answer
    logging.info('Best Clustering:')
    logging.info(predict_C)
    logging.info('k: {}'.format(epoch))
    # 指标评价
    F1_RESULT = metrics.f1_score(targets_id, predict_C, average='macro')
    logging.info('F1_: {}%'.format(F1_RESULT*100))
    acc_RESULT = metrics.accuracy_score(targets_id, predict_C)
    logging.info('acc: {}%'.format(acc_RESULT*100))
    NMI_RESULT = metrics.normalized_mutual_info_score(targets_id, predict_C, average_method='arithmetic')
    logging.info('NMI: {}%'.format(NMI_RESULT*100))
    ari_RESULT = ari_score(targets_id, predict_C)
    logging.info('ari: {}%'.format(ari_RESULT*100))



if __name__ == "__main__":
    main()
