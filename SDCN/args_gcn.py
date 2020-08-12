### CONFIGS ###
import torch
dataset = 'citeseer'
# dataset = 'cora'

if dataset=='citeseer':
    input_dim = 3703
if dataset=='cora':
    # input_dim = 1433
    input_dim = 1433
if dataset=='cora':
    cluster_num = 7
if dataset=='citeseer':
    cluster_num = 6
use_feature = True
cuda = torch.cuda.is_available()

learning_rate = 0.01
mixmode = False     # "Enable CPU GPU mixing mode."
lr = 0.02
weight_decay = 5e-4
type = 'densegcn'   # "Choose the model to be trained.(mutigcn, resgcn, densegcn, inceptiongcn)"
inputlayer = 'gcn'
outputlayer = 'gcn'
hidden1 = 128
hidden2 = 8
# hidden1 = 500
# hidden2 = 500
# hidden3 = 2000
# hidden4 = 10
dropout = 0
withbn = False    # 'Enable Bath Norm GCN'
withloop = False  # "Enable loop layer GCN"
nhiddenlayer = 1  # 'The number of hidden layers.'
nbaseblocklayer = 30 # =0  "The number of layers in each baseblock"
# nhiddenlayer = 6  # 'The number of hidden layers.'
# sampling_percent = 0.4  # "The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix."
# nbaseblocklayer = 18 # =0  "The number of layers in each baseblock"
aggrmethod = "default"  # "The aggrmethod for the layer aggreation. The options includes add and concat. Only valid in resgcn, densegcn and inecptiongcn"
if aggrmethod == "default":
    if type == "resgcn":
        aggrmethod = "add"
    else:
        aggrmethod = "concat"
if type == "mutigcn":
    print("For the multi-layer gcn model, the aggrmethod is fixed to nores and nhiddenlayers = 1.")
    nhiddenlayer = 1
    aggrmethod = "nores"