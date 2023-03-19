import argparse
import torch
from torch.optim import Adam, SGD
from generator import TrainGraph
from utils import calculate_loss
from model import DrBC

def train(model, lr, optimizer, batch_size, episodes, device):
    device = torch.device("cuda:0" if torch.cuda.is_available() and device == "gpu" else "cpu")
    model.to(device)
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)

    for iteration in range(episodes):

        # start training
        model.train()
        src, target = graph.get_source_target_pairs()

        outs = model(train_data, edge_index)
        loss = calculate_loss(outs, label, src, target)

        if iteration % LOG_INTERVAL == 0:
            print("[{}/{}] Loss:{:.4f}".format(iteration, TOTAL_ITERATION, loss.item()))

        loss.backward()
        optimizer.step()



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    # parsing the corresponding parameters for training
    lr = arg_parser.add_argument("--learning_rate", type=float, default=1e-4)
    optimizer = arg_parser.add_argument("--optimizer", type=str, default="adam")
    batch_size = arg_parser.add_argument("--batch_size", type=int, default=16)
    episodes = arg_parser.add_argument("--episode", type=int, default=10000)
    scale = arg_parser.add_argument("--scale", type=tuple, default=(100, 200))
    device = arg_parser.add_argument("--device", type=str, default="gpu")

    model = DrBC()
    train(model, lr, optimizer, batch_size, episodes, device)
