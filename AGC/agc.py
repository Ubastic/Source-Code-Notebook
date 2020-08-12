import numpy as np
import pandas as pd
import networkx as nx

import time
from scipy.sparse import linalg as la
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score as ari_score

import scipy.sparse as sp
import sys
import pickle as pkl
import logging

self_loop = False
normal_feature = False

def agc(nx_graph, adj, features, targets_id, classes, start_time, flag,stop_threshold = -0.06):
    '''
    Parameters
    ----------
    nx_graph:    network类型的图数据
    adj:         邻接矩阵
    features:    特征矩阵
    targets_id:  label
    classes:     类别数
    stop_threshold: 类间距变化的停止与阈值
    return
    ----------
    predict_C:  预测结果
    epoch:      epoch阶
    '''
    if flag==1:
        adj_matrix, feat_matrix = get_matrix(adj, features, self_loop=self_loop)
    else:
        adj_matrix = adj
        feat_matrix = features
        features = sp.csc_matrix(features)
        adj = sp.csc_matrix(adj)
    matrix_A = sp.csr_matrix(adj_matrix) 
    dim_A = matrix_A.shape       # (3312, 3312)

    # 转换度矩阵 matrix_D
    matrix_D = np.zeros(dim_A)
    degree_table = nx.degree(nx_graph) # len = 3312
    ind = 0
    for degree_item in nx.degree(nx_graph):
        if degree_item[0] in nx_graph.nodes:
            matrix_D[ind][ind] = degree_item[1]
            ind += 1
        else:
            logging.info('miss a item ->', degree_item)

    # 归一化拉普拉斯算子matrix_Ls
    #   求矩阵的-1/2次方
    #   v 为特征值    Q 为特征向量
    matrix_D = sp.csr_matrix(matrix_D)  
    # v, Q = la.eigs(matrix_D)
    # V = sp.diags(v**(-0.5))
    # matrix_D_neg_1_2 = Q.dot(V).dot(la.inv(sp.csr_matrix(Q)))
    matrix_D_neg_1_2 = sp.csr_matrix.power(matrix_D, -0.5)  
    matrix_Ls = np.identity(dim_A[0]) - matrix_D_neg_1_2.dot(matrix_A).dot(matrix_D_neg_1_2)

    # features单位化 影响不大
    if normal_feature:
        features_dense = features.todense()
        deno = np.repeat(np.sqrt(np.sum(np.multiply(features_dense, features_dense), axis = 1)), features_dense.shape[1], axis = 1)
        features_normal = np.multiply(features_dense, 1.0 / deno)

    x_hat = matrix_X = features.todense()   # features_normal
    coefficient = np.identity(dim_A[0]) - 1/2 * matrix_Ls

    max_iter = 140
    predict_C = []
    tmp_intra0 = tmp_intra1 = 1e8
    t = 0    
    while t <= max_iter:
        #计算时差
        logging.info("iter: " + str(t) + ", at: " + str(time.time() - start_time) + " s")

        t = t + 1
        # k = t
        # x_hat = (coefficient ** k).dot(matrix_X)

        x_hat = coefficient.dot(x_hat)
        matrix_K = x_hat.dot(x_hat.T)

        matrix_W = 1/2 * (np.abs(matrix_K) + np.abs(matrix_K.T))

        matrix_W = matrix_W / (np.max(matrix_W))
        label_pred = SpectralClustering(n_clusters=classes, 
                                        gamma=0, 
                                        affinity='precomputed', # 改输入为亲和矩阵
                                        n_init=15,
                                        n_jobs=4,
                                        # kernel_params= matrix_K,
                                        assign_labels='kmeans', # discretize
                                        ).fit_predict(matrix_W)
        print('label_pred', label_pred)
        print('targets_id', targets_id)

        # 换位簇配对
        confusion_matrix = np.zeros([classes, classes])
        tmp_dict0 = np.ones(targets_id.shape)
        tmp_label_pred = 100 * np.ones_like(label_pred)
        for i in range(classes):
            for j in range(classes):
                confusion_matrix[i,j] = np.sum(tmp_dict0[np.where((targets_id==i) * (label_pred==j))])
        # print(confusion_matrix)
        diag_max = np.sum(np.diag(confusion_matrix))
        
        tmp_inds = list(range(classes))

        for _ in range(5):
            for i in range(classes):
                for j in range(classes):
                    confusion_matrix[[j, i], :] = confusion_matrix[[i, j], :]
                    if np.sum(np.diag(confusion_matrix)) < diag_max:
                        confusion_matrix[[i, j], :] = confusion_matrix[[j, i], :]                 
                    else:
                        diag_max = np.sum(np.diag(confusion_matrix))
                        tmp_inds[i], tmp_inds[j] = tmp_inds[j], tmp_inds[i]
                        # logging.info(diag_max)

        for i in range(classes):
            tmp_label_pred[np.where(label_pred == i)] = tmp_inds[i]
        print('tmp_label_pred', tmp_label_pred)

        # 指标评价
        logging.info('k: {}'.format(t))
        # 指标评价
        F1_RESULT = metrics.f1_score(targets_id, tmp_label_pred, average='macro')
        logging.info('F1_: {}%'.format(F1_RESULT*100))
        acc_RESULT = metrics.accuracy_score(targets_id, tmp_label_pred)
        with open('agc_wiki_oversmooth.txt','a') as overfile:
            overfile.write(str(acc_RESULT))
            overfile.write('\n')
        logging.info('acc: {}%'.format(acc_RESULT*100))
        NMI_RESULT = metrics.normalized_mutual_info_score(targets_id, tmp_label_pred, average_method='arithmetic')
        logging.info('NMI: {}%'.format(NMI_RESULT*100))
        ari_RESULT = ari_score(targets_id, tmp_label_pred)
        logging.info('ari: {}%'.format(ari_RESULT * 100))

        tmp_intra0 = tmp_intra1
        tmp_intra1 = intra(tmp_label_pred, x_hat, classes)
        d_intra = tmp_intra1 - tmp_intra0
        logging.info('d_intra: {}'.format(d_intra))

        # if d_intra > stop_threshold:
        #     break

        predict_C = tmp_label_pred

    return predict_C, t-1


def get_matrix(adj, features, self_loop=False):
    adj = sp.coo_matrix(adj)    # 没变啊。。
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])   # 加自环
    adj_matrix = adj.toarray()

    feat_matrix = features.toarray()
    return adj_matrix, feat_matrix

def statistics_list(arr, classes):
    result = {}
    for i in range(classes):
        result[i] = arr.count(i)
    return result

def intra(label_pred, x_hat, classes):
    intra_c = []
    tmp_list = statistics_list(label_pred.tolist(), classes)
    n = len(tmp_list)
    
    for ind in range(n):

        N = sum(label_pred==ind)

        if N in [0, 1]:  # 没有就不用加这一项了
            continue
        x_hat_cut = x_hat[np.where(label_pred==ind)[0]]

        x1_2 = np.repeat(np.sum(np.power(x_hat_cut, 2), axis = 1), N, axis = 1)
        x2_2 = np.repeat(np.sum(np.power(x_hat_cut, 2), axis = 1).T, N, axis = 0)
        x1_x2 = x_hat_cut.dot(x_hat_cut.T)

        dist_2 = x1_2 + x2_2 - 2 * x1_x2
        dist_2[np.where(x1_2 + x2_2 - 2 * x1_x2 < 0)] = 0

        dist = np.power(dist_2, 1/2)
        
        intra_c.append(0.5/(N * (N - 1)) * dist.sum())
        # print('intra_c[ind]', intra_c[ind])

    intra_ck = sum(intra_c)

    return intra_ck/n
