#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:02:15 2019

@author: YuxuanLong
"""

import torch
import torch.nn as nn
import torch.sparse as sp
import numpy as np
import utils
from loss import Loss

def sparse_drop(feature, drop_out):
    tem = torch.rand((feature._nnz()))
    feature._values()[tem < drop_out] = 0
    return feature

class GCMC(nn.Module):
    def __init__(self, feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, use_side, out_dim, drop_out = 0.0):
        # feature_u (943,2625)
        # feature_v (1682,2625)
        # feature_dim 2625
        # all_M_u 5lists (5,943,1682)
        # all_M_v 5lists (5,1682,943)
        # side_feature_u (943,21)
        # side_feature_v (1682,19)
        # all_M ndarray (5,943,1682)
        # mask (943,1682) True\False
        # user_item_matrix_train (943,1682)
        # user_item_matrix_train (943,1682)
        # laplacian_u (943)
        # laplacian_v (1682)
        super(GCMC, self).__init__()
        ###To Do:
        #### regularization on Q
        
        self.drop_out = drop_out
        
        side_feature_u_dim = side_feature_u.shape[1]    # 943
        side_feature_v_dim = side_feature_v.shape[1]    # 1682
        self.use_side = use_side    # 此时为0

        self.feature_u = feature_u  # (943,2625)
        self.feature_v = feature_v  # (1682,2625)
        self.rate_num = rate_num    # 5
        
        self.num_user = feature_u.shape[0]  # 943
        self.num_item = feature_v.shape[1]  # 1682
        
        self.side_feature_u = side_feature_u    # (943,21)
        self.side_feature_v = side_feature_v    # (943,1682)
        
        self.W = nn.Parameter(torch.randn(rate_num, feature_dim, hidden_dim))   # feature_dim 2625  hidden_dim 5
        nn.init.kaiming_normal_(self.W, mode = 'fan_out', nonlinearity = 'relu')
        
        self.all_M_u = all_M_u  # (5,943,1682)
        self.all_M_v = all_M_v  # (5,1682,943)
        
        self.reLU = nn.ReLU()
        
        if use_side:
            self.linear_layer_side_u = nn.Sequential(*[nn.Linear(side_feature_u_dim, side_hidden_dim, bias = True), 
                                                       nn.BatchNorm1d(side_hidden_dim), nn.ReLU()])
            self.linear_layer_side_v = nn.Sequential(*[nn.Linear(side_feature_v_dim, side_hidden_dim, bias = True), 
                                                       nn.BatchNorm1d(side_hidden_dim), nn.ReLU()])
    
            self.linear_cat_u = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2 + side_hidden_dim, out_dim, bias = True), 
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])
            self.linear_cat_v = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2 + side_hidden_dim, out_dim, bias = True), 
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])    
        else:
            
            self.linear_cat_u = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2, out_dim, bias = True), 
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])
            # Sequential(
            #       (0): Linear(in_features=50, out_features=5, bias=True)
            #       (1): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #       (2): ReLU()
            # )
            self.linear_cat_v = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2, out_dim, bias = True), 
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])
            # Sequential(
            #       (0): Linear(in_features=50, out_features=5, bias=True)
            #       (1): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #       (2): ReLU()
            # )

#        self.linear_cat_u2 = nn.Sequential(*[nn.Linear(out_dim, out_dim, bias = True), 
#                                            nn.BatchNorm1d(out_dim), nn.ReLU()])
#        self.linear_cat_v2 = nn.Sequential(*[nn.Linear(out_dim, out_dim, bias = True), 
#                                            nn.BatchNorm1d(out_dim), nn.ReLU()])
    
        self.Q = nn.Parameter(torch.randn(rate_num, out_dim, out_dim))  # out_dim 5
        nn.init.orthogonal_(self.Q)
        
        
    def forward(self):
        # dropout
        feature_u_drop = sparse_drop(self.feature_u, self.drop_out) / (1.0 - self.drop_out)  # (943,2625)
        feature_v_drop = sparse_drop(self.feature_v, self.drop_out) / (1.0 - self.drop_out)  # (1682,2625)
        
        hidden_feature_u = []
        hidden_feature_v = []

        # self.W (5,2625,5) -> (rate_num, feature_dim, hidden_dim)
        W_list = torch.split(self.W, self.rate_num)
        W_flat = []
        for i in range(self.rate_num):
            Wr = W_list[0][i]   # (2625,5)
            M_u = self.all_M_u[i]   # (943,1682)
            M_v = self.all_M_v[i]   # (1682,943)
            hidden_u = sp.mm(feature_v_drop, Wr)    # (1682,2625) * (2625,5) -> (1682,5)
            hidden_u = self.reLU(sp.mm(M_u, hidden_u))  # (943,1682) * (1682,5)
            
            ### need to further process M, normalization
            hidden_v = sp.mm(feature_u_drop, Wr)    # (943,2625) * (2625,5) -> (943,5)
            hidden_v = self.reLU(sp.mm(M_v, hidden_v))  # (1682,943) * (943,5)

            
            hidden_feature_u.append(hidden_u)
            hidden_feature_v.append(hidden_v)
            
            W_flat.append(Wr)   # 每个分数评价下的参数矩阵的存储
            
        hidden_feature_u = torch.cat(hidden_feature_u, dim = 1)     # (943，25)
        hidden_feature_v = torch.cat(hidden_feature_v, dim = 1)     # (1682,25)
        W_flat = torch.cat(W_flat, dim = 1)     # (2625,25)


        cat_u = torch.cat((hidden_feature_u, torch.mm(self.feature_u, W_flat)), dim = 1)    # (943,50) = (943，25) + (943,2625) * (2625,25)
        cat_v = torch.cat((hidden_feature_v, torch.mm(self.feature_v, W_flat)), dim = 1)    # (1682,50) = (1682,25) + (1682,2625) * (2625,25)
        
        if self.use_side:
            side_hidden_feature_u = self.linear_layer_side_u(self.side_feature_u)
            side_hidden_feature_v = self.linear_layer_side_v(self.side_feature_v)    
            
            cat_u = torch.cat((cat_u, side_hidden_feature_u), dim = 1)
            cat_v = torch.cat((cat_v, side_hidden_feature_v), dim = 1)
        
        
        embed_u = self.linear_cat_u(cat_u)  # nn.linear -> (943,50) -> (943,5)
        embed_v = self.linear_cat_v(cat_v)  # nn.linear -> (1682,50) -> (1682,5)
        
        score = []
        Q_list = torch.split(self.Q, self.rate_num)     # self.Q = (5,5,5)  参数共享矩阵
        for i in range(self.rate_num):
            Qr = Q_list[0][i]   # (5,5)
            
            tem = torch.mm(torch.mm(embed_u, Qr), torch.t(embed_v))     # (943,5)*(5,5)*(5,1682) -> (943,1682)
            
            score.append(tem)
        return torch.stack(score)
    

        


