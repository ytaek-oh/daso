"""Reference: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/mse_loss.py"""  # noqa
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from yacs.config import CfgNode

from typing import Optional

from .utils import weighted_loss


def build_mse_loss(cfg: CfgNode, **kwargs) -> nn.Module:
    return MSELoss(**kwargs)


@weighted_loss
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Warpper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')


class MSELoss(nn.Module):
    """MSELoss.
    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction: str = 'mean', loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        *,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None
    ) -> Tensor:
        """Forward function of loss.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * mse_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss
