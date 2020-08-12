import torch
import torch.nn as nn
from torch.nn import Module
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
import numpy as np


# several models for recommendations

# RMSE
# SVD dim = 50 50 epoch RMSE = 0.931
# GNCF dim = 64 layer = [64,64,64] nn = [128,64,32,] 50 epoch RMSE = 0.916/RMSE =0.914
# NCF dim = 64 50 nn = [128,54,32] epoch 50 RMSE = 0.928

class SVD(Module):

    def __init__(self,userNum,itemNum,dim):
        super(SVD, self).__init__()
        self.uEmbd = nn.Embedding(userNum,dim)
        self.iEmbd = nn.Embedding(itemNum,dim)
        self.uBias = nn.Embedding(userNum,1)
        self.iBias = nn.Embedding(itemNum,1)
        self.overAllBias = nn.Parameter(torch.Tensor([0]))

    def forward(self, userIdx,itemIdx):
        uembd = self.uEmbd(userIdx)
        iembd = self.iEmbd(itemIdx)
        ubias = self.uBias(userIdx)
        ibias = self.iBias(itemIdx)

        biases = ubias + ibias + self.overAllBias
        prediction = torch.sum(torch.mul(uembd,iembd),dim=1) + biases.flatten()

        return prediction

class NCF(Module):

    def __init__(self,userNum,itemNum,dim,layers=[128,64,32,8]):
        super(NCF, self).__init__()
        self.uEmbd = nn.Embedding(userNum,dim)
        self.iEmbd = nn.Embedding(itemNum,dim)
        self.fc_layers = torch.nn.ModuleList()
        self.finalLayer = torch.nn.Linear(layers[-1],1)

        for From,To in zip(layers[:-1],layers[1:]):
            self.fc_layers.append(nn.Linear(From,To))

    def forward(self, userIdx,itemIdx):
        uembd = self.uEmbd(userIdx)
        iembd = self.iEmbd(itemIdx)
        embd = torch.cat([uembd, iembd], dim=1)
        x = embd
        for l in self.fc_layers:
            x = l(x)
            x = nn.ReLU()(x)

        prediction = self.finalLayer(x)
        return prediction.flatten()


class GNNLayer(Module):

    def __init__(self,inF,outF):

        super(GNNLayer,self).__init__()
        self.inF = inF
        self.outF = outF
        self.linear = torch.nn.Linear(in_features=inF,out_features=outF)
        self.interActTransform = torch.nn.Linear(in_features=inF,out_features=outF)

    def forward(self, laplacianMat,selfLoop,features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # laplacianMat L = D^-1(A)D^-1 # 拉普拉斯矩阵
        L1 = laplacianMat + selfLoop    # 归一化拉普拉斯矩阵(2625,2625)
        L2 = laplacianMat.cuda()    # 将数据变量放到GPU上(2625,2625)
        L1 = L1.cuda()              # 将数据变量放到GPU上(2625,2625)
        inter_feature = torch.sparse.mm(L2,features)        # inter_feature(2625,80)
        inter_feature = torch.mul(inter_feature,features)   # inter_feature(2625,80)

        inter_part1 = self.linear(torch.sparse.mm(L1,features))  # inter_part1(2625,80)
        inter_part2 = self.interActTransform(torch.sparse.mm(L2,inter_feature))     # inter_part2(2625,80)

        return inter_part1+inter_part2

class GCF(Module):

    def __init__(self,userNum,itemNum,rt,embedSize=100,layers=[100,80,50],useCuda=True):

        super(GCF,self).__init__()
        self.useCuda = useCuda  # 是否使用GPU默认为True
        self.userNum = userNum  # 943users
        self.itemNum = itemNum  # 1682items
        self.uEmbd = nn.Embedding(userNum,embedSize)    # user特征嵌入 uEmbd:Embedding(943,80)  词典的大小尺寸,嵌入向量的维度，即用多少维来表示一个符号
        self.iEmbd = nn.Embedding(itemNum,embedSize)    # item特征嵌入 iEmbd:Embedding(1682,80) 词典的大小尺寸,嵌入向量的维度，即用多少维来表示一个符号
        self.GNNlayers = torch.nn.ModuleList()      # 多层GCN
        self.LaplacianMat = self.buildLaplacianMat(rt) # sparse format  user 943 item 1682 LaplacianMat--(2625,2625)

        self.leakyRelu = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)
        # layers=[80,80,]
        self.transForm1 = nn.Linear(in_features=layers[-1]*(len(layers))*2,out_features=64)  # Linear(320,64)
        self.transForm2 = nn.Linear(in_features=64,out_features=32)     # Linear(64,32)
        self.transForm3 = nn.Linear(in_features=32,out_features=1)      # Linear(32,1)

        for From,To in zip(layers[:-1],layers[1:]):
            self.GNNlayers.append(GNNLayer(From,To))    # GNN(80,80)

    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)

    def buildLaplacianMat(self,rt):

        rt_item = rt['itemId'] + self.userNum
        uiMat = coo_matrix((rt['rating'], (rt['userId'], rt['itemId'])))    # (943,1682)--rating

        uiMat_upperPart = coo_matrix((rt['rating'], (rt['userId'], rt_item)))   # (943,2625)--rating
        uiMat = uiMat.transpose()   # (1682,943)--rating
        uiMat.resize((self.itemNum, self.userNum + self.itemNum))   # (1682,2625)--rating
        # uiMat_upperPart--(943,2625)
        # uiMat--(1682,2625)
        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.userNum+self.itemNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL

    def getFeatureMat(self):
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()

        userEmbd = self.uEmbd(uidx)     # uidx 943 userEmbd(943,80)
        itemEmbd = self.iEmbd(iidx)     # iidx 1682 itemEmbd(1682,80)
        features = torch.cat([userEmbd,itemEmbd],dim=0)  # features(2625,80)
        return features

    def forward(self,userIdx,itemIdx):
        # 2.1 Embedding Layer
        itemIdx = itemIdx + self.userNum
        userIdx = list(userIdx.cpu().data)      # batch 2048 userIdx
        itemIdx = list(itemIdx.cpu().data)      # batch 2048 itemIdx
        # gcf data propagation
        features = self.getFeatureMat()     # features(2625,80)
        finalEmbd = features.clone()        # finalEmbd(2625,80)
        # 2.2 Embedding Propagation Layer
        for gnn in self.GNNlayers:
            # 2.2.1 Message Construction and 2.2.2 Message Aggregation
            features = gnn(self.LaplacianMat,self.selfLoop,features)
            features = nn.ReLU()(features)
            finalEmbd = torch.cat([finalEmbd,features.clone()],dim=1)

        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]
        # 2.3 Prediction Layer
        embd = torch.cat([userEmbd,itemEmbd],dim=1)

        embd = nn.ReLU()(self.transForm1(embd))
        embd = self.transForm2(embd)
        embd = self.transForm3(embd)
        prediction = embd.flatten()

        return prediction

if __name__ == '__main__':
    from toyDataset.loaddata import load100KRatings

    rt = load100KRatings()
    userNum = rt['userId'].max()
    itemNum = rt['itemId'].max()

    rt['userId'] = rt['userId'] - 1
    rt['itemId'] = rt['itemId'] - 1
    gcf = GCF(userNum,itemNum,rt)
