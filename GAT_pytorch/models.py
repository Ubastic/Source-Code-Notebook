import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        # print('nfeat:',nfeat)   # 引文网络每个结点的特征向量:描述论文的单词词汇表 1433
        # print('nhid:',nhid)
        # print('nclass:',nclass) # 7类论文
        # print('dropout:',dropout)   # 0.6
        # print('alpha:',alpha)       # 0.2
        # print('nheads:',nheads)     # 8
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # 搭建8层GAT
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # print('self.add_module:',self.add_module)
        # 考虑周边8个点的注意力机制，特征维度1433->4
        # 输出相应的分类结果，由于综合考虑了周边八个点的注意力机制
        # 输入维度为4*8=32，最终目的的分类结果为7类
        # nhid * nheads 32
        # nclass    7
        # dropout   0.6
        # alpha     0.2
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # print('adj.shape:',adj.shape) # 2708*2708
        x = F.dropout(x, self.dropout, training=self.training)
        # 综合考虑周边八个点的维度拼接
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # ELU激活函数
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


# 系数
class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

