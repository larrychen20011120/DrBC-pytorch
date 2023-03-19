import random
import numpy as np
import torch
import networkx as nx

class TrainGraph:
    def __init__(self, batch_size, scale):
        # batch_size is for random generate
        self.batch_size = batch_size
        # synthesis the graph by networkx for every batch
        self.graphs = [
            _synthesis(scale) for _ in range(batch_size)
        ]

    # private method for training data to generate the graph
    # from paper it define different scales: scale is a tuple expressed for range
    def _synthesis(self, scale):
        node_number = random.randrange( scale[0], scale[1])
        return nx.powerlaw_cluster_graph(n=node_number, m=4, p=0.05)


class TestGraph:
    def __init__(self, graph_path, bc_path):
        # edges match two list (two nodes)
        self.edges = [ [], [] ]
        self.bc_values = []
        # read the graph txt data
        # '\t' as split and int values
        with open(graph_path, "r") as f:
            for line f.readlines():
                line = (line.rstrip()).split("\t")
                start_node, end_node = int(line[0]), int(line[1])
                # undirected graph (start->end) and (end->start)
                self.edges[0].append(start_node)
                self.edges[1].append(end_node)
                self.edges[0].append(end_node)
                self.edges[1].append(start_node)
        # read the score txt data
        # '\t' as split and (int value, float score)
        with open(bc_path, "r") as f:
            for line f.readlines():
                line = (line.rstrip()).split("\t")
                # node number matches the list index
                self.bc_values.append( float(line[1]) )
