import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class Encoder(MessagePassing):
    def __init__(self, input_size, embed_size, block_number):
        super(Encoder, self).__init__(aggr="add")

        self.block_number = block_number
        self.embed = nn.Linear(input_size, embed_size)
        self.relu = nn.ReLU(inplace=True)
        self.gru = nn.GRUCell(embed_size, embed_size)

    def forward(self, x, edge_idx):

        # compute the degrees and normalize terms
        row, col = edge_idx
        deg = degree(col)
        # in paper it uses sqrt(1 + degree) as normalization
        deg = torch.add(deg, 1)
        deg = torch.pow(deg, -0.5)
        norm = deg[row] * deg[col]
        # the first embedding
        x = self.embed(x)
        x = self.relu(x)
        # all the hidden layers will as the decoder's inputs
        hidden_layers = [x]

        for i in range(self.block_number):
            # in pyg propagate
            # it is the combination of update, message and aggregate
            x = self.propagate(edge_idx, x=x, norm=norm)
            hidden_layers.append(x)

        hidden_layers = torch.stack(hidden_layers, dim=-1)
        # max_pooling from all layers for decoder to decoding
        z, _ = torch.max(hidden_layers, dim=-1)

        return z

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, x):
        x = self.gru(x, aggr_out)
        return x


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
    def __init__(self, input_size=3, embed_size=128, block_number=5, feature_size=128, hidden_size=32):
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
    x = torch.tensor([
        [2, 1, 1],[2, 1, 1],[2, 1, 1],[2, 1, 1]
    ], dtype=torch.float)

    y = torch.tensor([
        [0, 3, 1, 2, 3, 2],[3, 0, 2, 1, 2, 3]
    ], dtype=torch.long)

    edge_idx = torch.tensor([
        [0,1,2], [1,2,3]
    ], dtype=torch.long)

    print(model(x, edge_idx))
