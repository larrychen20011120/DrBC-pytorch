import argparse

def train(model, lr, optimizer, episodes):
    pass

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    # parsing the corresponding parameters for training
    lr = argParser.add_argument("--learning_rate", type=float, default=1e-4)
    optimizer = argParser.add_argument("--optimizer", type=str, default="adam")
    episodes = argParser.add_argument("--episode", type=int, default=10000)
