from __future__ import print_function, division
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import *
from GNN import *
from evaluation import eva
import args_gcn
from collections import Counter
from tensorboardX import *


# torch.cuda.set_device(1)

writer = SummaryWriter()
class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)   # input->500
        self.enc_2 = Linear(n_enc_1, n_enc_2)   # 500->500
        self.enc_3 = Linear(n_enc_2, n_enc_3)   # 500->2000
        self.z_layer = Linear(n_enc_3, n_z)     # 2000->z

        self.dec_1 = Linear(n_z, n_dec_1)       # z->2000
        self.dec_2 = Linear(n_dec_1, n_dec_2)   # 2000->500
        self.dec_3 = Linear(n_dec_2, n_dec_3)   # 500->500
        self.x_bar_layer = Linear(n_dec_3, n_input)  # 500->input

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))          # input->500
        enc_h2 = F.relu(self.enc_2(enc_h1))     # 500->500
        enc_h3 = F.relu(self.enc_3(enc_h2))     # 500->2000
        z = self.z_layer(enc_h3)                # 2000->z

        dec_h1 = F.relu(self.dec_1(z))          # z->2000
        dec_h2 = F.relu(self.dec_2(dec_h1))     # 2000->500
        dec_h3 = F.relu(self.dec_3(dec_h2))     # 500->500
        x_bar = self.x_bar_layer(dec_h3)        # 500->input

        return x_bar, enc_h1, enc_h2, enc_h3, z


## hh
class GCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,n_clusters,
                 n_input, n_z):
        super(GCN, self).__init__()
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

    def forward(self, x,tra1,tra2,tra3,adj,z):
        h = self.gnn_1(x, adj)  # torch.Size([3327, 500])
        # print(tra1.shape)   # torch.Size([3327, 500])
        h = self.gnn_2(h+tra1, adj)  # torch.Size([3327, 500])
        # print(tra2.shape)   # torch.Size([3327, 500])
        h = self.gnn_3(h+tra2, adj)  # torch.Size([3327, 2000])
        # print(tra3.shape)   # torch.Size([3327, 2000])
        h = self.gnn_4(h+tra3, adj)  # torch.Size([3327, 10])
        # print(z.shape)      # torch.Size([3327, 10])
        h = self.gnn_5(h+z, adj, active=False)  # torch.Size([3327, 6])

        return h


## multi gcn
class mutil_GCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,n_clusters,
                 n_input, n_z , adj):
        super(mutil_GCN, self).__init__()
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        # self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_4 = GCNModel(nfeat=n_enc_3,
                 nhid=args_gcn.hidden1,
                 nclass=n_z,
                 nhidlayer=args_gcn.nhiddenlayer,
                 dropout=args_gcn.dropout,
                 baseblock=args_gcn.type,
                 inputlayer=args_gcn.inputlayer,
                 outputlayer=args_gcn.outputlayer,
                 nbaselayer=args_gcn.nbaseblocklayer,
                 activation=F.relu,
                 withbn=args_gcn.withbn,
                 withloop=args_gcn.withloop,
                 aggrmethod=args_gcn.aggrmethod,
                 mixmode=args_gcn.mixmode)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

    def forward(self, x,tra1,tra2,tra3,adj,z):
        h = self.gnn_1(x, adj)  # torch.Size([3327, 500])
        # print(tra1.shape)   # torch.Size([3327, 500])
        h = self.gnn_2(h+tra1, adj)  # torch.Size([3327, 500])
        # print(tra2.shape)   # torch.Size([3327, 500])
        h = self.gnn_3(h+tra2, adj)  # torch.Size([3327, 2000])
        # print(tra3.shape)   # torch.Size([3327, 2000])
        h = self.gnn_4(h+tra3, adj)  # torch.Size([3327, 10])
        # print(z.shape)      # torch.Size([3327, 10])
        h = self.gnn_5(h+z, adj, active=False)  # torch.Size([3327, 6])

        return h


class SDCN(nn.Module):

    def __init__(self, dataset,n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                adj,n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # original
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # gcn##
        # self.gcn = GCN(n_enc_1=n_enc_1,n_enc_2=n_enc_2,n_enc_3=n_enc_3,
        #                n_input=n_input,n_clusters=n_clusters,n_z=n_z)
        # self.gcn.load_state_dict(torch.load('data/pretrain/cite111.pkl', map_location='cpu'))
        # gcn##

        # # multi gcn##
        self.gcn = mutil_GCN(n_enc_1=n_enc_1,n_enc_2=n_enc_2,n_enc_3=n_enc_3,
                       n_input=n_input,n_clusters=n_clusters,n_z=n_z,adj=adj)
        # # multi gcn##

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

        # dataset
        self.dataset = dataset
    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        # GCN Module
        # h = self.gnn_1(x, adj)  # torch.Size([3327, 500])
        # h = self.gnn_2(h+tra1, adj)  # torch.Size([3327, 500])
        # h = self.gnn_3(h+tra2, adj)  # torch.Size([3327, 2000])
        # h = self.gnn_4(h+tra3, adj)  # torch.Size([3327, 10])
        # h = self.gnn_5(h+z, adj, active=False)  # torch.Size([3327, 6])
        ##
        h = self.gcn(x,tra1,tra2,tra3,adj,z)
        ##
        predict = F.softmax(h, dim=1)       # torch.Size([3327, 6])
        adj_re = dot_product_decode(h)
        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # x_bar为decoder的结果
        # q为t分布
        # predict为softmax后预测的种类
        # z为encoder的潜在特征表示
        return x_bar, q, predict, z, adj_re

    # hh
    # def pretrain(self,path=''):
    #     if path == '':
    #         pretrain_ae(self.ae,self.dataset)
    #     # load pretrain weights
    #     if path == '':
    #         self.ae.load_state_dict(torch.load(args.train_pretrain_path))
    #         print('load pretrained ae from', args.train_pretrain_path)
    #     else:
    #         self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
    #         print('load pretrained ae from', args.pretrain_path)

# hh
# def pretrain_ae(model,dataset):
#     '''
#     pretrain autoencoder
#     '''
#     print(model)
#     print('pretrain_ae')
#     optimizer = Adam(model.parameters(), lr=args.lr)
#     for epoch in range(1000):
#         total_loss = 0.
#         x = torch.Tensor(dataset.x).to(device)
#
#         optimizer.zero_grad()
#         x_bar,_,_,_, z = model(x)
#         loss = F.mse_loss(x_bar, x)
#         total_loss += loss.item()
#
#         loss.backward()
#         optimizer.step()
#
#         print("epoch {} loss={:.4f}".format(epoch,
#                                             total_loss))
#         torch.save(model.state_dict(), args.train_pretrain_path)
#     print("model saved to {}.".format(args.train_pretrain_path))


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset):
    # KNN Graph
    adj = load_graph(args.name, args.k)
    adj = adj.cuda()

    model = SDCN(dataset=dataset,
                 n_enc_1=500, n_enc_2=500, n_enc_3=2000,
                 n_dec_1=2000, n_dec_2=500, n_dec_3=500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0,adj=adj).to(device)
    # print(model)
    # model.pretrain(args.pretrain_path)
    if args.name == 'cite':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    if args.name == 'dblp':
        optimizer = Adam(model.parameters(), lr=args.lr)



    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')
    # ## Visualization
    # with SummaryWriter(comment='model') as w:
    #     testdata = torch.rand(3327,3703).cuda()
    #     testadj = torch.rand(3327,3327).cuda()
    #     w.add_graph(model,(testdata,testadj))
    # ##
    for epoch in range(200):
        if epoch % 1 == 0:
        # update_interval
            # x_bar为decoder的结果
            # q为t分布
            # predict为softmax后预测的种类
            # z为encoder的潜在特征表示
            _, tmp_q, pred, _ ,adj_re= model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            # eva(y, res1, str(epoch) + 'Q')
            acc,nmi,ari,f1 = eva(y, res2, str(epoch) + 'Z')
            ## Visualization
            # writer.add_scalar('checkpoints/scalar', acc, epoch)
            # writer.add_scalar('checkpoints/scalar', nmi, epoch)
            # writer.add_scalar('checkpoints/scalar', ari, epoch)
            # writer.add_scalar('checkpoints/scalar', f1, epoch)
            ##
            # eva(y, res3, str(epoch) + 'P')

        # x_bar为decoder的结果
        # q为t分布
        # predict为softmax后预测的种类
        # z为encoder的潜在特征表示

        x_bar, q, pred, _ ,adj_re= model(data, adj)


        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        # print('kl_loss:{}       ce_loss:{}'.format(kl_loss, ce_loss))
        # print(p.shape)            # torch.Size([3327, 6])
        # print(q.log().shape)      # torch.Size([3327, 6])
        # print(pred.log().shape)   # torch.Size([3327, 6])
        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss
        # adj_loss = norm*F.binary_cross_entropy(adj_re.view(-1).cpu(), adj_label.to_dense().view(-1).cpu(), weight = weight_tensor)
        # loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss + 0.1*adj_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='dblp')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='')
    parser.add_argument('--train_pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    # execute pretrain
    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    # args.train_pretrain_path = 'data/pretrain/{}.pkl'.format(args.name)
    dataset = load_data(args.name)

    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256
        args_gcn.nhiddenlayer = 3
        args_gcn.nbaseblocklayer = 12

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561
        args_gcn.nhiddenlayer = 3
        args_gcn.nbaseblocklayer = 12

    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000
        args_gcn.nhiddenlayer = 3
        args_gcn.nbaseblocklayer = 12

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334
        args_gcn.sampling_percent = 0.05

    if args.name == 'cite':
        args.lr = 2e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703

    if args.name == 'cora':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 7
        args.n_input = 1433


    print(args,'nhiddenlayer: ',args_gcn.nhiddenlayer,'nbaseblocklayer: ',args_gcn.nbaseblocklayer,'sampling_percent: ',args_gcn.sampling_percent)
    train_sdcn(dataset)
