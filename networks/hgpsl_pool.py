import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv

from .layers.hgpsl_pool import GCN, HGPSLPool


class Model(torch.nn.Module):
    def __init__(self, args, in_channels, num_classes):
        super(Model, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.nhid = args.nhid
        self.num_classes = num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.in_channels, self.nhid)
        # self.conv2 = GCN(self.nhid, self.nhid)
        # self.conv3 = GCN(self.nhid, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        # self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        # self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        # self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)
        self.lin = torch.nn.Linear(self.nhid * 2, self.num_classes)

    def forward(self, x, edge_index, batch):
        edge_attr = None

        x = F.relu(self.conv1(x.float(), edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.log_softmax(self.lin3(x), dim=-1)
        # x = self.lin3(x)

        # return x
        return self.lin(x)
