import torch
import numpy as np
import random

def fix_random_seeds(seed=0):
    """ Fix random seeds. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = dim

    def forward(self, predictions, labels, eval = False):
        predictions = predictions.log_softmax(dim=self.dim)
        
        with torch.no_grad():
            indicator = 1.0 - labels
            smooth_labels = torch.zeros_like(labels)
            smooth_labels.fill_(self.smoothing / (self.classes - 1))
            smooth_labels = labels * self.confidence + indicator * smooth_labels#lables->indicator

        return torch.mean(torch.sum(-smooth_labels.cuda(2) * predictions, dim=self.dim))
    