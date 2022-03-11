import torch
import torch.nn as nn
from torch import Tensor

from lib.utils import Meters
from typing import Tuple, Union


def accuracy(output: Tensor, target: Tensor, topk: Tuple[int] = (1, )) -> Union[int, Tuple[int]]:
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
            res.append(correct_k.mul_(100.0 / batch_size).item())
        if len(res) == 1:
            return res[0]
        return res


class Accuracy(nn.Module):

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.classwise_acc = Meters()

    def forward(self, output: Tensor, target: Tensor, *, log_classwise: bool = False,
                prefix="") -> Union[int, Tuple[int]]:
        if log_classwise:
            # log classwise accuracy
            for class_idx in range(self.num_classes):
                cls_inds = torch.where(target == class_idx)[0]
                if len(cls_inds):
                    cls_acc = accuracy(output[cls_inds], target[cls_inds])
                    metric_key = f"{class_idx}"
                    if prefix:
                        metric_key = f"{prefix}_{metric_key}"
                    self.classwise_acc.put_scalar(metric_key, cls_acc, n=len(cls_inds))
        return accuracy(output, target, (1, 5))

    @property
    def classwise(self) -> dict:
        return self.classwise_acc.get_latest_scalars_with_avg()

    def reset_classwise_accuracy(self) -> None:
        self.classwise_acc.reset()
