### Module file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.data.batch import Batch

class GNN7L_Sage(nn.Module):
    def __init__(self, data, hidden_channels=16):
        super().__init__()
        self.conv1 = SAGEConv(data.num_features, hidden_channels, aggr='max')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='max')
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='max')
        self.conv4 = SAGEConv(hidden_channels, hidden_channels, aggr='max')
        self.conv5 = SAGEConv(hidden_channels, hidden_channels, aggr='max')
        self.conv6 = SAGEConv(hidden_channels, hidden_channels, aggr='max')
        self.conv7 = SAGEConv(hidden_channels, int(data.num_classes), aggr='max')
    
    def arguments_read(self, *args, **kwargs):

        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=x.device)
            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(args[0].shape[0], dtype=torch.int64, device=x.device)
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        return x, edge_index, batch
    
    def forward(self, x=None, edge_index=None, *args, **kwargs):
        if x == None and edge_index == None:
            x, edge_index, batch = self.arguments_read(*args, **kwargs)
            

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = F.relu(self.conv6(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv7(x, edge_index)

        return F.log_softmax(x, dim=1)