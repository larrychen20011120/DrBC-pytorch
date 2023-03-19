import argparse
from generator import TrainGraph
from model import DrBC

def train(model, lr, optimizer, batch_size, episodes):
    pass

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    # parsing the corresponding parameters for training
    lr = arg_parser.add_argument("--learning_rate", type=float, default=1e-4)
    optimizer = arg_parser.add_argument("--optimizer", type=str, default="adam")
    batch_size = arg_parser.add_argument("--batch_size", type=int, default=16)
    episodes = arg_parser.add_argument("--episode", type=int, default=10000)
    scale = arg_parser.add_argument("--scale", type=tuple, default=(100, 200))
