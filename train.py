import argparse
import os
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
        "acc_early": -100
    }
    # the step for checking early stopping
    train_step = 50
    # entry directory of the scale data
    entry = f'train_val_gen/{scale[0]}_{scale[1]}/'

    # device setting and model structure
    device = torch.device("cuda:0" if torch.cuda.is_available() and device == "gpu" else "cpu")
    print(model)
    model.to(device)

    # which optimizer to use => Adam or SGD
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)

    # training progress...
    for epoch in range(episodes):
        print(f"========================================== Epoch {epoch+1} ==========================================")
        # load batch data
        total_loss = 0
        for batch_path in os.listdir(os.path.join(entry, 'train')):
            train_graph = TrainGraph(batch_size=batch_size, scale=scale, path=os.path.join(entry, 'train', batch_path))
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
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Training loss ===> {total_loss/10000:.4f}")
        logs["train_loss"].append(total_loss/10000)

        # evaluation
        loss, acc = eval(model, entry, scale, device)
        logs["eval_loss"].append(loss)
        logs["eval_acc"].append(acc)
        if loss < logs["best_loss"]:
            logs["best_loss"] = loss
            logs["best_acc"] = acc
            torch.save(model.state_dict(), f"model_result/{scale[0]}_{scale[1]}.pth")
            print("Saving the best model weights!!")

        # Early stop (check every 500 iterations)
        # dump logs
        if epoch % train_step == 0:
            # should more than one percent improvement
            if acc-logs['acc_early'] < 1:
                return
            else:
                acc_early = acc
                # dump the log file
                with open(logs["path"], "wb") as f:
                    pickle.dump(logs, f)

def eval(model, entry, scale, device):
    model.eval()
    total_loss, total_acc = 0, 0
    for batch_path in os.listdir(os.path.join(entry, 'val')):
        eval_graph = TrainGraph(batch_size=1, scale=scale, path=os.path.join(entry, 'val', batch_path))
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
            total_loss += loss.item()
            metrics = Metrics()
            metrics.set_output(pred, gt)
            total_acc += metrics.top_k(k=10)

    print(f"Evaluation loss ===> {total_loss/20:.4f}")
    print(f"Evaluation Top-10% Acc ===> {total_acc/20:.2f}%")

    return total_loss, total_acc


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
