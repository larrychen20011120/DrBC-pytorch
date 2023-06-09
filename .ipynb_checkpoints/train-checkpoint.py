import argparse
import os
import pickle
import torch
from torch.optim import Adam, SGD
from generator import TrainGraph
from utils import calculate_loss, Metrics
from model import DrBC
import time
from torch_geometric.nn import Node2Vec

def train(model, lr, optimizer, batch_size, episodes, scale, device, early_step,):
    logs = {
        "path": f"logs/{scale[0]}_{scale[1]}.pkl",
        "train_loss": [],
        "eval_loss": [],
        "eval_acc": [],
        "best_loss": 1e8,
        "best_acc": 0,
        "acc_early": -100,
        "time": [time.time()]
    }
    # the step for checking early stopping
    early_step = early_step
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
            graph_input = train_graph.get_input()
            edge_idx = train_graph.get_edge_idx()
            gt = train_graph.get_ground_truth()
            src, target = train_graph.select_pairs()

            # start training
            model.train()
            # converting the data into torch tensor and certain device
            graph_input = torch.tensor(graph_input, dtype=torch.float)
            edge_idx = torch.tensor(edge_idx, dtype=torch.long)
            gt = torch.tensor(gt).view(-1, 1)

            # calculate loss and update parameters
            pred = model(graph_input.to(device), edge_idx.to(device))
            loss = calculate_loss(
                pred.to(device), gt.to(device), src, target
            )
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Training loss ===> {total_loss/len(os.listdir(os.path.join(entry, 'train'))):.10f}")
        logs["train_loss"].append(total_loss/len(os.listdir(os.path.join(entry, 'train'))))

        # evaluation
        loss, acc = eval(model, entry, scale, device)
        logs["eval_loss"].append(loss)
        logs["eval_acc"].append(acc)
        if acc > logs["best_acc"]:
            logs["best_loss"] = loss
            logs["best_acc"] = acc
            torch.save(model.state_dict(), f"model_result/{scale[0]}_{scale[1]}.pth")
            print("Saving the best model weights!!")

        # Early stop (check every 50 iterations)
        # dump logs
        if epoch % early_step == 0:
            # should more than one percent improvement
            logs['time'].append(time.time())
            if abs(acc-logs['acc_early']) < 1:
                print("Early stopping!!")
                return
            else:
                logs['acc_early'] = acc
                # dump the log file
                with open(logs["path"], "wb") as f:
                    pickle.dump(logs, f)

def eval(model, entry, scale, device):
    model.eval()
    total_loss, total_acc = 0, 0
    for batch_path in os.listdir(os.path.join(entry, 'val')):
        eval_graph = TrainGraph(batch_size=1, scale=scale, path=os.path.join(entry, 'val', batch_path))
        # preparing the data
        graph_input = eval_graph.get_input()
        edge_idx = eval_graph.get_edge_idx()
        gt = eval_graph.get_ground_truth()
        src, target = eval_graph.select_pairs()
        # converting the data into torch tensor and certain device
        graph_input = torch.tensor(graph_input, dtype=torch.float)
        edge_idx = torch.tensor(edge_idx, dtype=torch.long)
        gt = torch.tensor(gt).view(-1, 1)

        with torch.no_grad():
            pred = model(graph_input.to(device), edge_idx.to(device))
            loss = calculate_loss(
                pred.to(device), gt.to(device), src, target
            )
            total_loss += loss.item()
            metrics = Metrics()
            metrics.set_output(pred, gt)
            total_acc += metrics.top_k(k=10)

    print(f"Evaluation loss ===> {total_loss/100:.10f}")
    print(f"Evaluation Top-10% Acc ===> {total_acc/100:.2f}%")

    return total_loss, total_acc


def train_Node2Vec(scale):
    edge_idx = []
    for batch_path in os.listdir(os.path.join(entry, 'train')):
        train_graph = TrainGraph(batch_size=batch_size, scale=scale, path=os.path.join(entry, 'train', batch_path))
        edge_idx.append(train_graph.get_edge_idx())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(
        torch.tensor(edge_idx, dtype=torch.long), embedding_dim=128, p=1, q=2
    ).to(device)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        
        for pos_rw, neg_rw in tqdm(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    # parsing the corresponding parameters for training
    arg_parser.add_argument("--lr", type=float, default=1e-4)
    arg_parser.add_argument("--optimizer", type=str, default="adam")
    arg_parser.add_argument("--batch_size", type=int, default=16)
    arg_parser.add_argument("--episode", type=int, default=2000)
    arg_parser.add_argument("--scale", nargs='+', type=int)
    arg_parser.add_argument("--device", type=str, default="gpu")
    arg_parser.add_argument("--early_step", type=int, default=250)
    arg_parser.add_argument("--model", type=str, default="DrBC")
    args = arg_parser.parse_args()
    if (args.model == "DrBC"):
        model = DrBC()
        train(model, args.lr, args.optimizer, args.batch_size, args.episode, tuple(args.scale), args.device, early_step=args.early_step)
    else:
        train_Node2Vec(tuple(args.scale))
