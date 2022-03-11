import torch.nn as nn


class RotationHead(nn.Module):

    def __init__(self, in_features: int):
        super(RotationHead, self).__init__()
        self.projections = nn.Linear(in_features, 4)
        self._init_weights()

    def _init_weights(self):
        for m in self.projections.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.projections(x)
