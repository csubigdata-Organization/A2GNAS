from copy import deepcopy
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
import torch_geometric.utils as tg_utils
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GraphConv
import copy
from torch_geometric.nn import GINConv
from torch_geometric.nn import VGAE
from torch_geometric.utils import subgraph
from A2GNAS_core.search_space.propagation import NewGCNConv,NewGraphConv,NewGATConv,NewSAGEConv,NewGINConv,NewGeneralConv
from A2GNAS_core.search_space.local_pooling import TopKPool,SAGPool,ASAPool,PANPool,HopPool,GAPPool,NonePool


def attribute_masking(data, aug_ratio=0.2):
    data = deepcopy(data)

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    token = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)),
                             dtype=torch.float32).to(data.x.device)
    data.x[idx_mask] = token
    return data


def edge_perturbation(data, aug_ratio=0.2):
    data = deepcopy(data)

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    idx_add = np.random.choice(node_num, (2, permute_num))
    idx_drop = np.random.choice(edge_num, edge_num - permute_num, replace=False)

    edge_index = data.edge_index[:, idx_drop]
    data.edge_index = edge_index

    return data


def node_dropping(data, aug_ratio=0.2):
    data = deepcopy(data)

    x = data.x
    edge_index = data.edge_index

    drop_num = int(data.num_nodes * aug_ratio)
    keep_num = data.num_nodes - drop_num

    keep_idx = torch.randperm(data.num_nodes)[:keep_num]
    edge_index, edge_attr = tg_utils.subgraph(keep_idx, edge_index)

    drop_idx = torch.ones(x.shape[0], dtype=bool)
    drop_idx[keep_idx] = False
    x[drop_idx] = 0

    data.x = x
    data.edge_index = edge_index
    data.edge_attr = edge_attr

    return data


class GIN_NodeWeightEncoder(torch.nn.Module):
    def __init__(self, num_features, dim, add_mask=False):
        super().__init__()

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        if add_mask == True:
            ### 3 is True/False/Mask
            nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 3))
            self.conv2 = GINConv(nn2)
            self.bn2 = torch.nn.BatchNorm1d(3)
        else:
            ### 2 is True/False
            nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 2))
            self.conv2 = GINConv(nn2)
            self.bn2 = torch.nn.BatchNorm1d(2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        return x

class ViewGenerator(VGAE):
    def __init__(self, args, add_mask=False):

        self.num_features = args.num_features
        self.hidden_dim = args.hidden_dim
        self.add_mask = add_mask

        encoder = GIN_NodeWeightEncoder(self.num_features, self.hidden_dim, self.add_mask)
        super().__init__(encoder=encoder)

    def forward(self, data_in, requires_grad=True):
        data = copy.deepcopy(data_in)

        x, edge_index = data.x, data.edge_index
        edge_attr = None
        if data.edge_attr is not None:
            edge_attr = data.edge_attr

        data.x = data.x.float()
        x = x.float()
        x.requires_grad = requires_grad

        p = self.encoder(data)
        sample = F.gumbel_softmax(p, hard=True)

        real_sample = sample[:, 0]
        attr_mask_sample = None
        if self.add_mask == True:
            attr_mask_sample = sample[:, 2]
            keep_sample = real_sample + attr_mask_sample
        else:
            keep_sample = real_sample

        keep_idx = torch.nonzero(keep_sample, as_tuple=False).view(-1, )
        edge_index, edge_attr = subgraph(keep_idx, edge_index, edge_attr, num_nodes=data.num_nodes)
        x = x * keep_sample.view(-1, 1)

        if self.add_mask == True:
            attr_mask_idx = attr_mask_sample.bool()
            token = data.x.detach().mean()
            x[attr_mask_idx] = token

        data.x = x
        data.edge_index = edge_index
        if data.edge_attr is not None:
            data.edge_attr = edge_attr

        return data


def aug_map(aug_type, args):
    if aug_type == 'attribute_masking':
        return attribute_masking
    elif aug_type == 'edge_perturbation':
        return edge_perturbation
    elif aug_type == 'node_dropping':
        return node_dropping
    elif aug_type == 'autogcl':
        return ViewGenerator(args)
    else:
        raise Exception("wrong augmentation function")


def global_pooling_map(global_pooling_type):
    if global_pooling_type == 'global_max':
        global_pooling = global_max_pool
    elif global_pooling_type == 'global_mean':
        global_pooling = global_mean_pool
    elif global_pooling_type == 'global_add':
        global_pooling = global_add_pool
    else:
        raise Exception("Wrong pooling function")
    return global_pooling


def local_pooling_map(local_pooling_type, hidden_dim):
    if local_pooling_type == 'TopKPool':
        local_pooling = TopKPool(hidden_dim, min_score=-1)
    elif local_pooling_type == 'SAGPool':
        local_pooling = SAGPool(hidden_dim, min_score=-1, GNN=GCNConv)
    elif local_pooling_type == 'ASAPool':
        local_pooling = ASAPool(hidden_dim)
    elif local_pooling_type == 'PANPool':
        local_pooling = PANPool(hidden_dim, min_score=-1)
    elif local_pooling_type == 'HopPool':
        local_pooling = HopPool(hidden_dim)
    elif local_pooling_type == 'GCPool':
        local_pooling = SAGPool(hidden_dim, min_score=-1, GNN=GraphConv)
    elif local_pooling_type == 'GAPPool':
        local_pooling = GAPPool(hidden_dim)
    elif local_pooling_type == 'None':
        local_pooling = NonePool(hidden_dim)
    else:
        raise Exception("Wrong local pooling function")
    return local_pooling


### different propagation mechanisms based on different conv
def propagation_map(propa_type, input_dim, hidden_dim):
    if propa_type == 'GCN_PM':
        propa_layer = NewGCNConv(input_dim, hidden_dim, aggr='add')
    elif propa_type == 'GAT_PM':
        heads = 2
        propa_layer = NewGATConv(input_dim, hidden_dim//heads, heads=heads, aggr='add')
    elif propa_type == 'SAGE_PM':
        propa_layer = NewSAGEConv(input_dim, hidden_dim, normalize=True, aggr='add')
    elif propa_type == 'GIN_PM':
        nn1 = Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        propa_layer = NewGINConv(nn=nn1, aggr='add')
    elif propa_type == 'Graph_PM':
        propa_layer = NewGraphConv(input_dim, hidden_dim, aggr='add')
    elif propa_type == 'General_PM':
        propa_layer = NewGeneralConv(input_dim, hidden_dim, skip_linear=True, aggr='add')
    else:
        raise Exception("Wrong propagation mechanism")
    return propa_layer
