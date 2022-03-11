import torch
import torch.nn as nn


class Queue(nn.Module):

    def __init__(self, max_size=128, device=None):
        super(Queue, self).__init__()
        self.max_size = max_size
        self._bank = []
        self.device = device

    def enqueue(self, features: torch.Tensor):
        with torch.no_grad():
            self._bank.append(features)
        current_size = len(self._bank)
        if current_size > self.max_size:
            self._bank = self._bank[-self.max_size:]
        assert len(self._bank) <= self.max_size

    def forward(self, features):
        self.enqueue(features)
        return torch.cat(self._bank, 0).mean(0).to(self.device)


class DistributionAlignment(nn.Module):

    def __init__(self, cfg, target_dist, eps=1e-6):
        super().__init__()
        self.target_dist = target_dist
        self.eps = eps

        self.da_t = cfg.MODEL.DIST_ALIGN.TEMPERATURE  # default temperature value
        self.avg_pred = Queue(device=cfg.GPU_ID)

    def set_target_dist(self, target_dist):
        self.target_dist = target_dist

    def forward(self, p, temperature=None):
        with torch.no_grad():
            da_t = self.da_t if temperature is None else temperature
            avg_p = self.avg_pred(p).view(1, -1)

            target_dist = torch.pow(self.target_dist, da_t)
            target_dist = target_dist / target_dist.sum()

            p_align = p * (target_dist.view(1, -1) + self.eps) / (avg_p + self.eps)

            return p_align / p_align.sum(dim=1, keepdim=True)
