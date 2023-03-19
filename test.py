import argparse
from generator import TestGraph
from model import DrBC

def test():
    pass

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    graph_path = arg_parser.add_argument("--graph_path", type=str, required=True)
    bc_path = arg_parser.add_argument("--bc_path", type=str, required=True)
    model_path = arg_parser.add_argument("--model_path", type=str, default="model.pth")
    test_graph = TestGraph(graph_path=graph_path, bc_path=bc_path)
