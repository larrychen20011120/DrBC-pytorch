import argparse
import pickle
import torch
from torch.optim import Adam, SGD
from generator import TrainGraph
from utils import calculate_loss, Metrics
from model import DrBC

def train(model, lr, optimizer, batch_size, episodes, scale, device):
    logs = {
        "path": f"logs/{scale[0]}_{scale[1]}.pkl",
        "train_loss": [],
        "eval_loss": [],
        "eval_acc": [],
        "best_loss": 1e8,
        "best_acc": 0,
        "acc_early": 0
    }
    train_step = 500
    train_graph = None
    eval_graph = TrainGraph(batch_size=1, scale=(4000,5000))

    device = torch.device("cuda:0" if torch.cuda.is_available() and device == "gpu" else "cpu")
    model.to(device)
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)

    # training progress...
    for iteration in range(episodes):
        print(f"========================================== Iteration {iteration+1} ==========================================")
        # generate graph every iteration
        if iteration % train_step == 0:
            train_graph = TrainGraph(batch_size=batch_size, scale=scale)
            print("Training: Completely generating the graph data")
        # preparing data
        input = train_graph.get_input()
        edge_idx = train_graph.get_edge_idx()
        gt = train_graph.get_ground_truth()
        src, target = train_graph.select_pairs()

        # start training
        model.train()
        # converting the data into torch tensor and certain device
        input = torch.tensor(input, dtype=torch.float)
        edge_idx = torch.tensor(edge_idx, dtype=torch.long)
        gt = torch.tensor(gt).view(-1, 1)

        # calculate loss and update parameters
        pred = model(input.to(device), edge_idx.to(device))
        loss = calculate_loss(
            pred.to(device), gt.to(device), src, target
        )
        loss.backward()
        optimizer.step()
        print(f"Training loss ===> {loss.item():.4f}")
        logs["train_loss"].append(loss)

        # evaluation
        loss, acc = eval(model, eval_graph, device)
        print(f"Evaluation Top-1% Acc ===> {acc:.2f}%")
        logs["eval_loss"].append(loss)
        logs["eval_acc"].append(acc)
        if loss < logs["best_loss"]:
            logs["best_loss"] = loss
            logs["best_acc"] = acc
            torch.save(model.state_dict(), "model.pth")
            print("Saving the best model weights!!")
        # dump the log file
        with open(logs["path"], "wb") as f:
            pickle.dump(logs, f)
        # Early stop (check every 500 iterations)
        if iteration % train_step == 0:
            if acc-logs['acc_early'] < 0.5:
                return
            else:
                acc_early = acc

def eval(model, eval_graph, device):
    model.eval()
    print("Evaluation: Completely generating the graph data")
    # preparing the data
    input = eval_graph.get_input()
    edge_idx = eval_graph.get_edge_idx()
    gt = eval_graph.get_ground_truth()
    src, target = eval_graph.select_pairs()
    # converting the data into torch tensor and certain device
    input = torch.tensor(input, dtype=torch.float)
    edge_idx = torch.tensor(edge_idx, dtype=torch.long)
    gt = torch.tensor(gt).view(-1, 1)

    with torch.no_grad():
        pred = model(input.to(device), edge_idx.to(device))
        loss = calculate_loss(
            pred.to(device), gt.to(device), src, target
        )
        print(f"Evaluation loss ===> {loss.item():.4f}")

    metrics = Metrics()
    metrics.set_output(pred, gt)
    return loss, metrics.top_k()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    # parsing the corresponding parameters for training
    arg_parser.add_argument("--lr", type=float, default=1e-4)
    arg_parser.add_argument("--optimizer", type=str, default="adam")
    arg_parser.add_argument("--batch_size", type=int, default=16)
    arg_parser.add_argument("--episode", type=int, default=10000)
    arg_parser.add_argument("--scale", nargs='+', type=int)
    arg_parser.add_argument("--device", type=str, default="gpu")
    args = arg_parser.parse_args()
    model = DrBC()
    train(model, args.lr, args.optimizer, args.batch_size, args.episode, tuple(args.scale), args.device)
