import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops


class Encoder(MessagePassing):
    def __init__(self, input_size, embed_size, block_number):
        super(Encoder, self).__init__(aggr="add")

        self.embed = nn.Linear(input_size, embed_size)
        self.grus  = nn.ModuleList([
            nn.GRU(embed_size, embed_size) for _ in range(block_number)
        ])
        self.relu_in = nn.ReLU(inplace=True)

    def forward(self, x, edge_idx):

        # the first embedding
        x = self.embed(x)
        x = self.relu_in(x)
        x = F.normalize(x, p=2, dim=1)
        # all the hidden layers will as the decoder's inputs
        hidden_layers = [x]

        for gru in self.grus:
            # compute the degrees and normalize terms
            # for neighboring aggregate
            # edge_idx, _ = add_self_loops(edge_idx, num_nodes=x.size(0))
            row, col = edge_idx
            deg = degree(col, x.size(0))
            # in paper it uses sqrt(1 + degree) as normalization
            deg = torch.add(deg, 1)
            deg = torch.pow(deg, -0.5)
            norm = deg[row] * deg[col]
            # in pyg propagate
            # it is the combination of update, message and aggregate
            h = self.propagate(edge_idx, x=x, norm=norm, size=(x.size(0),x.size(0)))
            h = h.view(1, *h.shape)
            x = x.view(1, *x.shape)
            h = gru(h, x)
            h = F.normalize(h[0][0], p=2, dim=1)
            hidden_layers.append(h)
            x = h

        hidden_layers = torch.stack(hidden_layers)
        # max_pooling from all layers for decoder to decoding
        z = torch.max(hidden_layers, dim=0).values

        return z

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class Decoder(nn.Module):
    # it is a simple MLP
    def __init__(self, feature_size, hidden_size):
        super().__init__()
        self.hidden = nn.Linear(feature_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, z):
        z = self.hidden(z)
        z = self.relu(z)
        z = self.output(z)
        return z


class DrBC(nn.Module):
    def __init__(self, input_size=3, embed_size=128, block_number=5, feature_size=128, hidden_size=64):
        super().__init__()
        self.encoder = Encoder(input_size, embed_size, block_number)
        self.decoder = Decoder(feature_size, hidden_size)

    def forward(self, x, edge_idx):
        z = self.encoder(x, edge_idx)
        z = self.decoder(z)
        return z

if __name__ == "__main__":
    # testing for the model result
    model = DrBC()
    x = torch.tensor([[2, 1, 1],[2, 1, 1],[2, 1, 1],[2, 1, 1]],dtype=torch.float)

    y = torch.tensor([[0,2,1,0,3],[3,1,0,1,2]],dtype=torch.long)

    edge_index = torch.tensor([[0,1,2],
                               [1,2,3]],dtype=torch.long)

    print(model(x, edge_index))
