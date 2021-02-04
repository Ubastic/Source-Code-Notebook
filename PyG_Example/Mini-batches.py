from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset


# batch_size是图的个数，而不是节点的个数
TUD_dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(TUD_dataset, batch_size=32, shuffle=True)

for batch in loader:
    print("batch:", batch)
    print("batch.y", batch.y)
    print("batch.num_graphs:", batch.num_graphs)

cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
loader = DataLoader(cora_dataset, batch_size=32, shuffle=True)
# Cora只有一个图，所以只循环了一次
for batch in loader:
    print("batch:", batch)
    print("batch.y", batch.y)
    print("batch.num_graphs:", batch.num_graphs)