import random
import numpy as np
import torch
import networkx as nx
import pickle
import argparse
from tqdm import tqdm
import math

class TrainGraph:
    def __init__(self, batch_size, scale, path=None):
        # batch_size is for random generate
        self.batch_size = batch_size
        self.scale = scale
        self.graphs, self.ground_truths = [], []
        if path is None:
            # synthesis the graph by networkx for every batch
            for _ in range(batch_size):
                graph, bc = self._synthesis(scale)
                self.graphs.append(graph)
                # ground_truth is in a line
                self.ground_truths.extend(bc)
        else: # load from the previous generation
            self._load(path)

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

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"graphs": self.graphs, "ground_truth": self.ground_truths}, f)
    def _load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.graphs, self.ground_truths = data['graphs'], data['ground_truth']

class TestGraph:
    def __init__(self, graph_path, bc_path):
        # edges match two list (two nodes)
        self.edges = [ [], [] ]
        self.bc_values = []
        # read the graph txt data
        with open(graph_path, "r") as f:
            for line in f.readlines():
                line = (line.rstrip()).split()
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
                line = (line.rstrip()).split()
                # node number matches the list index
                self.bc_values.append( float(line[1]) )

    # for training input
    def get_input(self):
        # initialize the node without edges to degree 0
        degrees = [0] * len(self.bc_values)
        # focus on the first node is OK
        # assume it is an undirected graph
        for edge in self.edges:
            degrees[edge[0]] += 1
        return [[degree, 1, 1] for degree in degrees]
        
    def get_edge_idx(self):
        return self.edges
    def get_ground_truth(self):
        return self.bc_values


## here is for generating the training data and evaluation data
if __name__ == "__main__":
    # parser of arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--batch_size", type=int, default=16)
    arg_parser.add_argument("--train_size", type=int, default=10000)
    arg_parser.add_argument("--eval_size", type=int, default=100)
    # index from 0~4
    arg_parser.add_argument("--scale", nargs='+', type=int)
    args = arg_parser.parse_args()

    # parameter settings
    scales = [
        (100, 200), (200, 300), (1000,1200), (2000, 3000), (4000, 5000)
    ]
    batch_size = args.batch_size
    train_size = args.train_size
    eval_size = args.eval_size
    scales = [ scales[idx] for idx in tuple(args.scale)]

    # generate the dataset
    # use tqdm for tracking the progress
    for scale in scales:
        number = int(train_size/batch_size)
        for i in tqdm(range(number)):
            train_data = TrainGraph(batch_size=batch_size, scale=scale)
            train_data.save(f"train_val_gen/{scale[0]}_{scale[1]}/train/{i}.pkl")
        for i in tqdm(range(eval_size)):
            val_data = TrainGraph(batch_size=1, scale=scale)
            val_data.save(f"train_val_gen/{scale[0]}_{scale[1]}/val/{i}.pkl")
