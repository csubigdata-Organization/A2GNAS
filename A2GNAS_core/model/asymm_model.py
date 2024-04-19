import torch
import torch.nn.functional as F
import torch.nn as nn
from A2GNAS_core.search_space.search_space_utils import aug_map, propagation_map, global_pooling_map, local_pooling_map


class AsymmModel(torch.nn.Module):

    def __init__(self,
                 sample_architecture,
                 args):

        super(AsymmModel, self).__init__()

        self.sample_architecture = sample_architecture
        self.args = args
        self.original_feature_num = self.args.num_features
        self.hidden_dimension = self.args.hidden_dim
        self.num_classes = self.args.num_classes

        self.aug_method = aug_map(self.sample_architecture[0], self.args)

        self.pre_processing = torch.nn.Linear(self.original_feature_num, self.hidden_dimension)

        self.propa_type = self.sample_architecture[1]
        self.local_pooling_type = self.sample_architecture[2]
        self.global_pooling_type = self.sample_architecture[3]
        self.ps_cell_architecture = [self.propa_type, self.local_pooling_type, self.global_pooling_type]

        self.PS_operators = int(self.sample_architecture[-2])
        self.AS_operators = int(self.sample_architecture[-1])
        self.weight_shared_cells_num = self.PS_operators + self.AS_operators

        self.weight_shared_cells = torch.nn.ModuleList()
        for i in range(self.weight_shared_cells_num):
            cell = PS_Cell(self.ps_cell_architecture, self.hidden_dimension, self.hidden_dimension, self.args)
            self.weight_shared_cells.append(cell)

        self.post_processing = nn.Sequential(nn.Linear(self.hidden_dimension, self.hidden_dimension),
                                             nn.ReLU(),
                                             nn.Linear(self.hidden_dimension, self.num_classes))
        self.init_global_pooling = global_pooling_map(self.global_pooling_type)


    def forward(self, data, augment=False):
        if augment:
            data = self.aug_method(data)
            depth = self.PS_operators + self.AS_operators
        else:
            depth = self.PS_operators

        reprs = []

        data.edge_weight = torch.ones(data.edge_index.size()[1], device=data.edge_index.device).float()
        data.x = F.relu(self.pre_processing(data.x))
        reprs.append(self.init_global_pooling(data.x, data.batch))

        for i in range(depth):
            cell = self.weight_shared_cells[i]
            data, r = cell(data)
            reprs.append(r)

        reprs_tensor = torch.stack(reprs, dim=-1)
        embed = reprs_tensor.sum(dim=-1)

        prediction = self.post_processing(embed)
        logits = F.log_softmax(prediction, dim=-1)
        return logits


class PS_Cell(nn.Module):
    def __init__(self, cell_architecture, in_features, hidden_dimension, args):
        super().__init__()
        self.cell_architecture = cell_architecture
        self.in_features = in_features
        self.hidden_dimension = hidden_dimension
        self.args = args

        self.propa = propagation_map(self.cell_architecture[0], self.in_features, self.hidden_dimension)
        self.local_pooling = local_pooling_map(self.cell_architecture[1], self.hidden_dimension)
        self.global_pooling = global_pooling_map(self.cell_architecture[2])

    def forward(self, data):
        data.x = self.propa(data.x, data.edge_index, edge_weight=data.edge_weight)

        data.x = F.relu(data.x)

        # local_pooling
        data.x, data.edge_index, data.edge_weight, data.batch = self.local_pooling(data.x, data.edge_index, batch=data.batch, edge_weight=data.edge_weight)[:4]

        # global_pooling
        global_graph_emb = self.global_pooling(data.x, data.batch)

        return data, global_graph_emb
