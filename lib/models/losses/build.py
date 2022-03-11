import torch.nn as nn
from yacs.config import CfgNode

from typing import List

from .cross_entropy import build_cross_entropy
from .ldam_loss import build_ldam_loss
from .mse_loss import build_mse_loss


def build_loss(
    cfg: CfgNode, loss_type: str, class_count: List[int] = None, class_weight=None, **kwargs
) -> nn.Module:
    if loss_type == "CrossEntropyLoss":
        return build_cross_entropy(cfg, class_weight=class_weight, **kwargs)
    elif loss_type == "MSELoss":
        return build_mse_loss(cfg, **kwargs)

    elif loss_type == "LDAMLoss":
        return build_ldam_loss(cfg, class_count, class_weight=class_weight)
    else:
        raise ValueError("{} is not available for loss type.".format(loss_type))
