import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):   # 底层节点的参数，feature的个数；隐层节点个数；最终的分类数
        super(GCN, self).__init__()     #  super()._init_()在利用父类里的对象构造函数

        self.gc1 = GraphConvolution(nfeat, nhid)        # gc1输入尺寸nfeat，输出尺寸nhid
        self.gc2 = GraphConvolution(nhid, nclass)       # gc2输入尺寸nhid，输出尺寸ncalss
        self.dropout = dropout

    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
    def forward(self, x, adj):
        # print('x0的维度:', x.shape)
        x = F.relu(self.gc1(x, adj))        # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        # print('x1的维度:', x.shape)
        x = F.dropout(x, self.dropout, training=self.training)      # x要dropout
        # print('x2的维度:', x.shape)
        x = self.gc2(x, adj)
        # print('x3的维度:', x.shape)
        return F.log_softmax(x, dim=1)
