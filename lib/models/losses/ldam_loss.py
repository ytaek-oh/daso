"""source code: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_ldam_loss(cfg, class_count, class_weight=None, **kwargs):
    return LDAMLoss(class_count, class_weight=class_weight, device=cfg.GPU_ID)


class LDAMLoss(nn.Module):
    # paper check: class_weight?

    def __init__(self, class_count, max_m=0.5, class_weight=None, s=30, device=None):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(class_count))
        m_list = m_list * (max_m / torch.max(m_list))
        self.m_list = m_list

        assert s > 0
        self.s = s
        self.class_weight = class_weight
        self.device = device

    def forward(self, x, target, reduction=None):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float().to(self.device)

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.class_weight)
