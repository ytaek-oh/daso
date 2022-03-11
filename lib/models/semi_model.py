import torch.nn as nn
from yacs.config import CfgNode

from .classifier import Classifier
from .heads.rotation_head import RotationHead
from .wrn import build_wrn


class SemiModel(nn.Module):
    """backbone + projection head"""

    def __init__(self, cfg: CfgNode):
        super(SemiModel, self).__init__()
        with_rotation_head = cfg.MODEL.WITH_ROTATION_HEAD

        # feature extractor
        self.encoder = build_wrn(cfg)
        num_classes = self.encoder.num_classes
        out_features = self.encoder.out_features

        # classifier
        self.classifier = Classifier(out_features, num_classes)
        self.projection = None
        if with_rotation_head:
            self.projection = RotationHead(out_features)

        # misc
        self.num_classes = num_classes
        self.out_features = out_features

    def forward(
        self,
        x,
        is_train=True,
        rotation=False,
        classification_mode=None,
        return_features=False
    ):
        x = self.encoder(x)
        if return_features:
            return x

        if (not is_train) or (self.projection is None):
            return self.classifier(x)

        if classification_mode is not None:
            assert classification_mode in ["linear", "rotation"]
        else:
            classification_mode = "linear"

        if classification_mode == "linear":
            return self.classifier(x)
        if classification_mode == "rotation":
            return self.projection(x)
