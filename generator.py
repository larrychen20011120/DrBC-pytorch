import random
import numpy as np
import torch
import networkx as nx

class TrainGraph:
    def __init__(self, batch_size, scale):
        # batch_size is for random generate
        self.batch_size = batch_size
        self.graphs, self.ground_truths = [], []
        # synthesis the graph by networkx for every batch
        for _ in range(batch_size):
            graph, bc = self._synthesis(scale)
            self.graphs.append(graph)
            # ground_truth is in a line
            self.ground_truths.extend(bc)

    # private method for training data to generate the graph
    # from paper it define different scales: scale is a tuple expressed for range
    def _synthesis(self, scale):
        node_number = random.randrange( scale[0], scale[1])
        graph = nx.powerlaw_cluster_graph(n=node_number, m=4, p=0.05)
        bc = list(nx.betweenness_centrality(graph).values())
        return graph, bc

    # private method for calculating degree
    def _calculate_degree(self):
        degrees = []
        for graph in self.graphs:
            # drop index value out
            degrees.extend(
                [value for _, value in nx.degree(graph)]
            )
        return degrees

    # for loss
    def select_pairs(self):
        pair_multiple = 5
        src, target = [], []
        start = 0      # used for  identifying the start index
        for graph in self.graphs:
            # each batch node numbers (accumation of start)
            node_number = len(graph.nodes)
            # 5|V| pairs for loss calculation
            before_shuffle = list(range(start, start+node_number)) * pair_multiple
            random.shuffle(before_shuffle)
            src.extend(before_shuffle)
            before_shuffle = list(range(start, start+node_number)) * pair_multiple
            random.shuffle(before_shuffle)
            target.extend(before_shuffle)
            start += node_number # for next batch to generate index
        return src, target

    def get_edge_idx(self):
        edge_idx = [[], []]  # node pairs on differnent dims
        start = 0      # used for  identifying the start index
        for graph in self.graphs:
            # each batch node numbers (accumation of start)
            node_number = len(graph.nodes)
            for begin_node, end_node in graph.edges:
                # undirected
                edge_idx[0].append(begin_node + start)
                edge_idx[1].append(end_node + start)
                edge_idx[0].append(end_node + start)
                edge_idx[1].append(begin_node + start)
            start += node_number # for next batch of graph
        return edge_idx

    # for training input
    def get_input(self):
        return [[degree, 1, 1] for degree in self._calculate_degree()]
    def get_ground_truth(self):
        return self.ground_truths

class TestGraph:
    def __init__(self, graph_path, bc_path):
        # edges match two list (two nodes)
        self.edges = [ [], [] ]
        self.bc_values = []
        # read the graph txt data
        # '\t' as split and int values
        with open(graph_path, "r") as f:
            for line in f.readlines():
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
            for line in f.readlines():
                line = (line.rstrip()).split("\t")
                # node number matches the list index
                self.bc_values.append( float(line[1]) )
