import torch
from torch_geometric.data import Data

# 获取无向图的相反的边
def getEdgeOPP(edge):
    edge_opp = []
    singular = []
    dual = []
    for i in range(len(edge)):

        if i%2 == 0:
            singular.append(edge[i])
        if i%2 == 1:
            dual.append(edge[i])
    for i in range(len(dual)):
        edge_opp.append(dual[i])
        edge_opp.append(singular[i])
    return edge_opp

r"""A plain old python object modeling a single graph with various
(optional) attributes:
Args:
    x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
        num_node_features]`. (default: :obj:`None`)
    edge_index (LongTensor, optional): Graph connectivity in COO format
        with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
    edge_attr (Tensor, optional): Edge feature matrix with shape
        :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
    y (Tensor, optional): Graph or node targets with arbitrary shape.
        (default: :obj:`None`)
    pos (Tensor, optional): Node position matrix with shape
        :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
    normal (Tensor, optional): Normal vector matrix with shape
        :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
    face (LongTensor, optional): Face adjacency matrix with shape
        :obj:`[3, num_faces]`. (default: :obj:`None`)

The data object is not restricted to these attributes and can be extented
by any other additional data.

Example::

    data = Data(x=x, edge_index=edge_index)
    data.train_idx = torch.tensor([...], dtype=torch.long)
    data.test_mask = torch.tensor([...], dtype=torch.bool)
"""
edge_index_1 = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
edge_index_2 = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
edge_index_2 = edge_index_2.t().contiguous()
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
edge = [0, 1, 1, 2, 2, 3, 3, 4]
edge_opp = getEdgeOPP(edge)
print("original edge:", edge)
print("processed edge:", edge_opp)
# 官方说明文档
# x: 二维数组[num_nodes,num_node_features]说明了节点个数和节点特征，即第一维代表节点个数，第二维代表节点特征维度
# edge_index：二维数组[2, num_edges]说明结点之间的连接关系，默认描述为有向图，对于无向图相反即可
# edge_attr：二维数组[num_edges,num_edge_features]说明了边的个数和边特征，即第一维代表边个数，第二维代表边特征维度
# y：二维数组[num_nodes,node_teargets]说明了节点个数和节点标签，即第一维代表节点个数，第二维代表节点标签

data_1 = Data(x=x, edge_index=edge_index_1, y=[0,1,2])
data_2 = Data(x=x, edge_index=edge_index_2, y=[0,1,2])
# contains_isolated_nodes() 判断是否含有孤立节点
# contains_self_loops() 判断是否含有自环
# is_directed() 判断是否有向
print("data_1:", data_1, "data_1.num_edges:", data_1.num_edges, "data_1.contains_isolated_nodes():", data_1.contains_isolated_nodes(), "data_1.contains_self_loops():", data_1.contains_self_loops(), "data_1.is_directed():", data_1.is_directed())
print("data_2:", data_2, "data_2.num_edges:", data_2.num_edges, "data_2.contains_isolated_nodes():", data_2.contains_isolated_nodes(), "data_2.contains_self_loops():", data_2.contains_self_loops(), "data_2.is_directed():", data_2.is_directed())
# GPU or CPU
device = torch.device('cuda')
data_1 = data_1.to(device)
data_2 = data_2.to(device)