import sys
sys.path.append('../')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from GNN import *
from utils import *
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva


#torch.cuda.set_device(3)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class GCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,
                 n_input, n_z):
        super(GCN, self).__init__()
        self.gnn_1 = GNNLayer_pretrain(n_input, n_enc_1)
        self.gnn_2 = GNNLayer_pretrain(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer_pretrain(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer_pretrain(n_enc_3, n_z)

    def forward(self, x,adj):
        # x = x.cpu()
        # adj = adj.cpu()
        h = self.gnn_1(x, adj)  # torch.Size([3327, 500])
        h = self.gnn_2(h, adj)  # torch.Size([3327, 500])
        h = self.gnn_3(h, adj)  # torch.Size([3327, 2000])
        h = self.gnn_4(h, adj)  # torch.Size([3327, 10])
        adj_pre = dot_product_decode(h)
        return adj_pre,h

def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    # print('labels_all')
    # print(labels_all.shape)
    preds_all = (adj_rec > 0.5).view(-1).long()
    # print('preds_all')
    # print(preds_all.shape)
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def pretrain_ae(model, dataset, y):
    # print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(1):
        # adjust_learning_rate(optimizer, epoch)
        # dataset.x = dataset.x.cuda()
        A_pred, embding = model(dataset.x,adj_norm)
        embding = embding.detach().numpy()
        print(embding.shape)
        kmeans = KMeans(n_clusters=3, random_state=0,n_init=20).fit(embding)
        predict_labels = kmeans.predict(embding)
        optimizer.zero_grad()
        loss = log_lik = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1),
                                                       weight=weight_tensor)
        loss.backward()
        optimizer.step()
        train_acc = get_acc(A_pred, adj_label)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(train_acc))
        cm = clustering_metrics(y, predict_labels)
        cm.evaluationClusterModelFromLabel()
    torch.save(model.state_dict(), 'pretrain/pubmed1.pkl')



model = GCN(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_input=500,
        n_z=10,)


x = np.loadtxt('pubmed.txt', dtype=float)

y = np.loadtxt('pubmed_label.txt', dtype=int)

## load data
dataset = LoadDataset(x)
adj_train,adj_norm = load_graph_pretrain('pubmed', None)
##
adj = adj_train
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                            torch.FloatTensor(adj_label[1]),
                            torch.Size(adj_label[2]))
dataset.x = dataset.x.astype('float32')
dataset.x = torch.from_numpy(dataset.x)
print(type(adj_norm),adj_norm.shape)
print(type(adj_label),adj_label.shape)
print(type(dataset.x),dataset.x.shape)
weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
weight_tensor[weight_mask] = pos_weight
## train
pretrain_ae(model, dataset, y)
