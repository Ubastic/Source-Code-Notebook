from torch_geometric.datasets import Planetoid

# 加载数据集，从Github下载，如果下载失败可能需要科学上网
cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
print("cora_dataset:", cora_dataset)
print("type(cora_dataset):", type(cora_dataset))

# 通过dataset[0]访问数据集，基本信息为edge_index，test_mask，train_mask，val_mask，x，y
cora_data = cora_dataset[0]
print("cora_data:", cora_data)     
# 判断是否为有向图   
print("cora_data.is_undirected():", cora_data.is_undirected())
# 用于训练、测试、验证的全部图节点的bool-tensor类型列表
print("cora_data.train_mask:", cora_data.train_mask, cora_data.train_mask.sum().item())
print("cora_data.val_mask:", cora_data.val_mask, cora_data.val_mask.sum().item())
print("cora_data.test_mask:", cora_data.test_mask, cora_data.test_mask.sum().item())