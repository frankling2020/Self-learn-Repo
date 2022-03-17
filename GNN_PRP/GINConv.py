import torch, torch_sparse, torch_geometric
from torch_geometric.nn import GINConv, MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import nn, Tensor
import torch.nn.functional as F

class GIN(MessagePassing):
    def __init__(self, in_dims:int, out_dims:int, eps:float = 0, train_eps:bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GIN, self).__init__(**kwargs)
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.nn = nn.Sequential(nn.Linear(in_dims, out_dims), nn.ReLU(), nn.Linear(out_dims, out_dims))
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
            
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.nn:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
        self.eps.data.fill_(self.initial_eps)
        
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: torch_sparse.SparseTensor, x:OptPairTensor) -> Tensor:
        return torch_sparse.matmul(adj_t, x[0], reduce=self.aggr)
    
    def forward(self, x:Tensor, edge_index:Adj, size:Size=None):
        out = self.propagate(edge_index, x=(x, x), size=size)
        out += (1+self.eps) * x
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
    