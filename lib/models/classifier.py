import torch.nn as nn
from torch import Tensor


class Classifier(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, *, bias: bool = True
    ) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)

    def _init_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
