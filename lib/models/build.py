import torch
from yacs.config import CfgNode

from .abc_model import ABCModel
from .daso_model import DASOModel
from .semi_model import SemiModel
from .usadtm_model import USADTMModel


def build_model(cfg: CfgNode) -> torch.nn.Module:
    if cfg.ALGORITHM.NAME == "DASO":
        model = DASOModel(cfg)
    elif cfg.ALGORITHM.NAME == "USADTM":
        model = USADTMModel(cfg)
    elif cfg.ALGORITHM.NAME == "FixMatchABC":
        model = ABCModel(cfg)
    else:
        model = SemiModel(cfg)
    return model
