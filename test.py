import argparse
from generator import TestGraph
from model import DrBC
from utils import Metrics
import torch

def test(model, test_graph, device):
    device = torch.device("cuda:0" if torch.cuda.is_available() and device == "gpu" else "cpu")
    metrics = Metrics()    

    print("Start testing!!")
    model.to(device)
    model.eval()
    # preparing the data
    graph_input = test_graph.get_input()
    edge_idx = test_graph.get_edge_idx()
    gt = test_graph.get_ground_truth()

    # converting the data into torch tensor and certain device
    graph_input = torch.tensor(graph_input, dtype=torch.float)
    edge_idx = torch.tensor(edge_idx, dtype=torch.long)
    gt = torch.tensor(gt).view(-1, 1)

    with torch.no_grad():
        metrics.start_timer()
        pred = model(graph_input.to(device), edge_idx.to(device))
        metrics.end_timer()
        metrics.set_output(pred, gt)

    print("=============================== Result ===============================")
    print(f"1. Running time for testing: {metrics.get_runtime()}")
    print(f"2. Top 1% Acc ===> {metrics.top_k(k=1):.2f}%")
    print(f"3. Top 5% Acc ===> {metrics.top_k(k=5):.2f}%")
    print(f"4. Top 10% Acc ===> {metrics.top_k(k=10):.2f}%")
    print(f"5. Kendall Tau Statistics ===> {metrics.kendall_tau():.2f}")
    print("======================================================================")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--graph_path", type=str, required=True)
    arg_parser.add_argument("--bc_path", type=str, required=True)
    arg_parser.add_argument("--model_path", type=str, default="model.pth")
    arg_parser.add_argument("--device", type=str, default="cpu")
    args = arg_parser.parse_args()
    print("Loading data...")
    test_graph = TestGraph(graph_path=args.graph_path, bc_path=args.bc_path)
    model = DrBC()
    model.load_state_dict(torch.load(args.model_path))
    test(model, test_graph, args.device)
