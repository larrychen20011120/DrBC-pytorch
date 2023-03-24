import argparse
from generator import TestGraph
from model import DrBC

def test(model, test_graph, device):
    metrics = Metrics()
    # start timer
    metrics.start_timer()

    print("Start testing!!")
    model.eval()
    # preparing the data
    input = test_graph.get_input()
    edge_idx = test_graph.get_edge_idx()
    gt = test_graph.get_ground_truth()

    # converting the data into torch tensor and certain device
    input = torch.tensor(input, dtype=torch.float)
    edge_idx = torch.tensor(edge_idx, dtype=torch.long)
    gt = torch.tensor(gt).view(-1, 1)

    with torch.no_grad():
        pred = model(input.to(device), edge_idx.to(device))
        metrics.set_output(pred, gt)

    metrics.end_timer()
    print(f"Running time for testing: {metrics.get_runtime()}")
    print(f"Top 1% Acc ===> {metrics.top_k(k=1):.2f}%")
    print(f"Top 5% Acc ===> {metrics.top_k(k=5):.2f}%")
    print(f"Top 10% Acc ===> {metrics.top_k(k=10):.2f}%")
    print(f"Kendall Tau Statistics ===> {metrics.kendall_tau():.2f}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    graph_path = arg_parser.add_argument("--graph_path", type=str, required=True)
    bc_path = arg_parser.add_argument("--bc_path", type=str, required=True)
    model_path = arg_parser.add_argument("--model_path", type=str, default="model.pth")
    print("Loading data...")
    test_graph = TestGraph(graph_path=graph_path, bc_path=bc_path)
