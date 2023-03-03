import torch
import torch.nn.functional as F

from utils import recur


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1, 1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def Perplexity(output, target):
    with torch.no_grad():
        # label_mask = torch.arange(output.size(1), device=output.device)[output.sum(dim=[0,2]) != 0]
        # label_map = output.new_zeros(output.size(1), device=output.device, dtype=torch.long)
        # output = output[:, label_mask,]
        # label_map[label_mask] = torch.arange(output.size(1), device=output.device)
        # target = label_map[target]
        ce = F.cross_entropy(output, target)
        perplexity = torch.exp(ce).item()
    return perplexity


class Metric(object):
    def __init__(self):
        self.metric = {'Loss': (lambda input, output, loss: loss.item()),
                       'Local-Loss': (lambda input, output, loss: loss.item()),
                       'Global-Loss': (lambda input, output, loss: loss.item()),
                       'Accuracy': (lambda input, output, loss: recur(Accuracy, output, input['label'])),
                       'Local-Accuracy': (lambda input, output, loss: recur(Accuracy, output, input['label'])),
                       'Global-Accuracy': (lambda input, output, loss: recur(Accuracy, output, input['label'])),
                       'Perplexity': (lambda input, output, loss: recur(Perplexity, output, input['label'])),
                       'Local-Perplexity': (lambda input, output, loss: recur(Perplexity, output, input['label'])),
                       'Global-Perplexity': (lambda input, output, loss: recur(Perplexity, output, input['label']))}

    def evaluate(self, metric_names, input, output, loss):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output, loss)
        return evaluation