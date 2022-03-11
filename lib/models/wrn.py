import torch.nn as nn
from yacs.config import CfgNode


def conv3x3(i_c, o_c, stride=1):
    return nn.Conv2d(i_c, o_c, 3, stride, 1, bias=False)


def relu():
    return nn.LeakyReLU(0.1)


class residual(nn.Module):

    def __init__(self, input_channels, output_channels, stride=1, activate_before_residual=False):
        super().__init__()
        layer = []
        if activate_before_residual:
            self.pre_act = nn.Sequential(nn.BatchNorm2d(input_channels), relu())
        else:
            self.pre_act = nn.Identity()
            layer.append(nn.BatchNorm2d(input_channels))
            layer.append(relu())
        layer.append(conv3x3(input_channels, output_channels, stride))
        layer.append(nn.BatchNorm2d(output_channels))
        layer.append(relu())
        layer.append(conv3x3(output_channels, output_channels))

        if stride >= 2 or input_channels != output_channels:
            self.identity = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
        else:
            self.identity = nn.Identity()

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        x = self.pre_act(x)
        return self.identity(x) + self.layer(x)


class WRN(nn.Module):
    """ WRN28-width with leaky relu (negative slope is 0.1)"""

    def __init__(self, width: int, num_classes: int = 10) -> None:
        super().__init__()

        self.init_conv = conv3x3(3, 16)

        filters = [16, 16 * width, 32 * width, 64 * width]

        unit1 = [residual(filters[0], filters[1], activate_before_residual=True)] + \
            [residual(filters[1], filters[1]) for _ in range(1, 4)]
        self.unit1 = nn.Sequential(*unit1)

        unit2 = [residual(filters[1], filters[2], 2)] + \
            [residual(filters[2], filters[2]) for _ in range(1, 4)]
        self.unit2 = nn.Sequential(*unit2)

        unit3 = [residual(filters[2], filters[3], 2)] + \
            [residual(filters[3], filters[3]) for _ in range(1, 4)]
        self.unit3 = nn.Sequential(*unit3)

        self.unit4 = nn.Sequential(*[nn.BatchNorm2d(filters[3]), relu(), nn.AdaptiveAvgPool2d(1)])
        # self.classifier = Classifier(filters[3], num_classes, cls_cfg)

        self.num_classes = num_classes
        self.out_features = filters[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        f = self.unit4(x)
        # c = self.classifier(f.squeeze())
        return f.squeeze()


def build_wrn(cfg: CfgNode) -> nn.Module:
    # fmt: off
    width = cfg.MODEL.WIDTH
    num_classes = cfg.MODEL.NUM_CLASSES
    # fmt: on
    return WRN(width, num_classes=num_classes)
