import time
import torch
from torch.nn import BCEWithLogitsLoss as BCELoss
import scipy.stats as stats


def calculate_loss(pred, gt, src, target):
    # loss = -[ sig(b)*log(sig(y)) + sig(1-b)*log(sig(1-y)) ]
    # https://zhuanlan.zhihu.com/p/449140733
    # in BCEWithLogitsLoss the prediction will automatically apply sigmoid function
    # transform the ground_truth to (0,1) by sigmoid function
    loss = BCELoss()
    gt = torch.sigmoid(gt)
    return loss( pred[src]-pred[target], gt[src]-gt[target] )

class Metrics:
    def __init__(self):
        self.timer = 0
        self.prediction = None
        self.ground_truth = None

    def set_output(self, pred, gt):
        self.prediction, self.ground_truth = pred, gt

    def top_k(self, k=1): # in this paper use 1, 5, 10%
        # align two list and match them in accuracy
        pred = self.prediction.reshape(-1)
        gt = self.ground_truth.reshape(-1)
        # use torch topk choose top k's value
        k_node = int( k/100 * len(gt))
        _, pred_idx = torch.topk(pred, k_node)

        _, gt_idx = torch.topk(gt, k_node)
        # transform to numpy therefore can use set
        # cuda can't turn into numpy
        pred_set, gt_set = set(pred_idx.cpu().numpy()), set(gt_idx.cpu().numpy())
        # return the acc in percentage
        return len(pred_set & gt_set) / k_node * 100

    def kendall_tau(self):
        # use stats kendall_tau
        pred = self.prediction.reshape(-1)
        gt = self.ground_truth.reshape(-1)
        # cuda can't turn into numpy
        tau, _ = stats.kendalltau(pred.cpu().numpy(), gt.cpu().numpy())

    # runtime calculation
    def start_timer(self):
        self.timer = -time.time()
    def end_timer(self):
        self.timer += time.time()
    def get_runtime(self): # in seconds
        return int(self.timer)
