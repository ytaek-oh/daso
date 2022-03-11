from copy import deepcopy

import torch.nn as nn


class EMAModel(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        ema_decay: float,
        ema_weight_decay: float,
        device: str,
        *,
        resume: str = None
    ):
        super().__init__()
        # init model
        ema_model = deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)

        self.ema_model = ema_model
        self.ema_decay = ema_decay
        self.ema_weight_decay = ema_weight_decay

        self.train()

    def update(self, model, step, current_lr):
        ema_decay = min(1 - 1 / (step + 1), self.ema_decay)  # EMA warmup

        # parameter update
        for emp_p, p in zip(self.ema_model.parameters(), model.parameters()):
            emp_p.data = ema_decay * emp_p.data + (1 - ema_decay) * p.data

        # buffer update (i.e., running mean in BN)
        for emp_p, p in zip(self.ema_model.buffers(), model.buffers()):
            emp_p.data = ema_decay * emp_p.data + (1 - ema_decay) * p.data

        # EMA model weight decay
        self.apply_weight_decay(current_lr)
        return ema_decay

    def apply_weight_decay(self, current_lr):
        for m in self.ema_model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.data *= (1.0 - current_lr * self.ema_weight_decay)

    def forward(self, x, return_features=False, **kwargs):
        if return_features:
            return self.ema_model.encoder(x)
        return self.ema_model(x, **kwargs)

    def train(self):
        self.ema_model.train()

    def eval(self):
        self.ema_model.eval()
