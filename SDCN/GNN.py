import math
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from layers import *


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output


class GNNLayer_pretrain(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer_pretrain, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = glorot_init(self.in_features, self.out_features)
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        if active:
            output = F.relu(output)
        return output

device = torch.device("cuda:0")
class GCNModel(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False):
        """
        Initial function.
        :param nfeat: the input feature dimension.  # 1433
        :param nhid:  the hidden feature dimension. # 128
        :param nclass: the output feature dimension.    # 7
        :param nhidlayer: the number of hidden blocks.  # 1
        :param dropout:  the dropout ratio.         # 0.5
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".    # resgcn
        :param inputlayer: the input layer type, can be "gcn", "dense", "none". # gcn
        :param outputlayer: the input layer type, can be "gcn", "dense".        # gcn
        :param nbaselayer: the number of layers in one hidden block.    # 6
        :param activation: the activation function, default is ReLu.    # relu
        :param withbn: using batch normalization in graph convolution.  # False
        :param withloop: using self feature modeling in graph convolution.  # False
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".    # add
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.    # False
        """
        super(GCNModel, self).__init__()
        self.mixmode = mixmode
        self.dropout = dropout

        if baseblock == "resgcn":
            self.BASEBLOCK = ResGCNBlock
        elif baseblock == "densegcn":
            self.BASEBLOCK = DenseGCNBlock
        elif baseblock == "mutigcn":
            self.BASEBLOCK = MultiLayerGCNBlock
        elif baseblock == "inceptiongcn":
            self.BASEBLOCK = InecptionGCNBlock
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))
        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
            baseblockinput = nhid
        elif inputlayer == "none":
            self.ingc = lambda x: x
            baseblockinput = nfeat
        else:
            self.ingc = Dense(nfeat, nhid, activation)
            baseblockinput = nhid

        outactivation = lambda x: x
        if outputlayer == "gcn":
            self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)
        # elif outputlayer ==  "none": #here can not be none
        #    self.outgc = lambda x: x
        else:
            self.outgc = Dense(nhid, nclass, activation)

        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        # for i in xrange(nhidlayer):
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 out_features=nhid,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)

        self.reset_parameters()
        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)

    def reset_parameters(self):
        pass

    def forward(self, fea, adj):
        # input
        if self.mixmode:
            x = self.ingc(fea, adj.cpu())
        else:
            x = self.ingc(fea, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        if self.mixmode:
            x = x.to(device)

        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(x, adj)
        # output, no relu and dropput here.
        x = self.outgc(x, adj)
        x = F.log_softmax(x, dim=1)
        return x


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)

