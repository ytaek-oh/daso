"""Reference: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/cross_entropy_loss.py"""  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from yacs.config import CfgNode

from typing import List, Optional, Tuple

from .utils import weight_reduce_loss


def build_cross_entropy(cfg: CfgNode, loss_weight: float = 1.0, class_weight=None) -> nn.Module:
    use_sigmoid = cfg.MODEL.LOSS.CROSSENTROPY.USE_SIGMOID
    return CrossEntropyLoss(
        use_sigmoid=use_sigmoid, loss_weight=loss_weight, class_weight=class_weight
    )


def cross_entropy(
    pred: Tensor,
    label: Tensor,
    *,
    with_activation: bool = False,
    weight: Optional[Tensor] = None,
    reduction: str = 'mean',
    avg_factor: Optional[int] = None,
    class_weight: Optional[List[float]] = None,
    **kwargs
) -> Tensor:
    """Calculate the CrossEntropy loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
    Returns:
        torch.Tensor: The calculated loss
    """
    if label.ndim > 1:
        # logits as prediction and one hot labels
        assert pred.ndim == label.ndim
        loss = -1 * torch.sum(F.log_softmax(pred, dim=1) * label, dim=1)  # (N, )
    else:
        if with_activation:
            loss = F.nll_loss(pred.log(), label, weight=class_weight, reduction="none")
        else:
            loss = F.cross_entropy(pred, label, weight=class_weight, reduction="none")

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_binary_labels(labels: Tensor, label_weights: Optional[Tensor],
                          label_channels: int) -> Tuple[Tensor]:
    bin_labels = F.one_hot(labels, num_classes=label_channels)

    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(
    pred: Tensor,
    label: Tensor,
    *,
    with_activation=False,
    weight: Optional[Tensor] = None,
    reduction: str = 'mean',
    avg_factor: Optional[int] = None,
    class_weight: Optional[List[float]] = None
) -> Tensor:
    """Calculate the binary CrossEntropy loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
    Returns:
        torch.Tensor: The calculated loss
    """
    assert not with_activation
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight=class_weight, reduction='none'
    )
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


class CrossEntropyLoss(nn.Module):

    def __init__(
        self,
        use_sigmoid: bool = False,
        reduction: str = 'mean',
        class_weight: Optional[List[float]] = None,
        loss_weight: Optional[float] = 1.0
    ) -> None:
        """CrossEntropyLoss.
        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(
        self,
        cls_score: Tensor,
        label: Tensor,
        *,
        with_activation: bool = False,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
        class_weight_override=None,
        **kwargs
    ) -> Tensor:
        """Forward function.
        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = self.class_weight
            if isinstance(class_weight, list):
                class_weight = cls_score.new_tensor(class_weight)
        else:
            class_weight = None
        if class_weight_override is not None:
            class_weight = class_weight_override
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            with_activation=with_activation,
            weight=weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss_cls
