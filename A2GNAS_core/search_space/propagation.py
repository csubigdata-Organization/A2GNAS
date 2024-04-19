from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul, set_diag
from torch_geometric.nn import GCNConv, GraphConv, GATConv, SAGEConv, GINConv, GeneralConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class NewGCNConv(GCNConv):
    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        ### transform
        x = self.lin(x)


        ### propagate
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                 size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class NewGraphConv(GraphConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        ### propagate
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)

        ### transform
        out = self.lin_rel(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_root(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)


class NewGATConv(GATConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, edge_weight: OptTensor = None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        ### transform
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_src(x).view(-1, H, C)
            alpha_l = alpha_r = (x_l * self.att_src).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_src(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_src).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_dst(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_dst).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None
        self.add_self_loops = False
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                num_nodes = size[1] if size is not None else num_nodes
                num_nodes = x_r.size(0) if x_r is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        ### propagate
        out = self.propagate(edge_index, x=(x_l, x_r),
                                 alpha=(alpha_l, alpha_r), size=size, edge_weight=edge_weight)
        alpha = self._alpha
        self._alpha = None


        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias


        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, edge_weight: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        x1 = x_j * alpha.unsqueeze(-1)
        if edge_weight is None:
            return x1
        else:
            x2 = (x1.view(-1, self.heads * self.out_channels).t() * edge_weight).t().view(-1, self.heads, self.out_channels)
            return x2


class NewSAGEConv(SAGEConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        ### propagate
        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)

        ### transform
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class NewGINConv(GINConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        ### propagate
        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)


        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        ### transform
        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class NewGeneralConv(GeneralConv):

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        x_self = x[1]
        ### propagate
        out = self.propagate(edge_index, x=x, size=size,
                                 edge_weight=edge_weight)
        out = out.mean(dim=1)  # todo: other approach to aggregate heads

        ### transform
        out += self.lin_self(x_self)
        if self.normalize_l2:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message_basic(self, x_i: Tensor, x_j: Tensor, edge_weight: Tensor):
        if self.directed_msg:
            x_j = self.lin_msg(x_j)
        else:
            x_j = self.lin_msg(x_j) + self.lin_msg_i(x_i)
        if edge_weight is not None:
            x_j = edge_weight.view(-1, 1) * x_j
        return x_j

    def message(self, x_i: Tensor, x_j: Tensor, edge_index_i: Tensor,
                size_i: Tensor, edge_weight: Tensor) -> Tensor:
        x_j_out = self.message_basic(x_i, x_j, edge_weight)
        x_j_out = x_j_out.view(-1, self.heads, self.out_channels)
        if self.attention:
            if self.attention_type == 'dot_product':
                x_i_out = self.message_basic(x_j, x_i, edge_weight)
                x_i_out = x_i_out.view(-1, self.heads, self.out_channels)
                alpha = (x_i_out * x_j_out).sum(dim=-1) / self.scaler
            else:
                alpha = (x_j_out * self.att_msg).sum(dim=-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
            alpha = alpha.view(-1, self.heads, 1)
            return x_j_out * alpha
        else:
            return x_j_out
