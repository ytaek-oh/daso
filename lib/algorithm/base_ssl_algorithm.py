import math

import torch.nn as nn
from yacs.config import CfgNode

from lib.models import EMAModel
from lib.models.dist_align import DistributionAlignment
from lib.models.losses import build_loss

from .base_algorithm import BaseAlgorithm
from .darp_reproduce import DARP


class SemiSupervised(BaseAlgorithm):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        self.resume = None
        self.ema_model = EMAModel(
            self.model,
            cfg.MODEL.EMA_DECAY,
            cfg.MODEL.EMA_WEIGHT_DECAY,
            device=self.device,
            resume=self.resume
        )
        self.ul_loss = self.build_unlabeled_loss(cfg)
        self.apply_scheduler = cfg.SOLVER.APPLY_SCHEDULER

        # confidence threshold for unlabeled predictions in PseudoLabel and FixMatch algorithms
        self.conf_thres = cfg.ALGORITHM.CONFIDENCE_THRESHOLD

        # distribution alignment
        self.with_align = cfg.MODEL.DIST_ALIGN.APPLY
        if self.with_align:
            self.dist_align = DistributionAlignment(cfg, self.p_data)

        # apply darp
        self.with_darp = cfg.ALGORITHM.DARP.APPLY
        if self.with_darp:
            ul_dataset = self.ul_loader.dataset
            self.darp_optimizer = DARP(cfg, ul_dataset)

    def build_unlabeled_loss(self, cfg: CfgNode) -> nn.Module:
        loss_type = cfg.MODEL.LOSS.UNLABELED_LOSS
        loss_weight = cfg.MODEL.LOSS.UNLABELED_LOSS_WEIGHT

        ul_loss = build_loss(cfg, loss_type, class_count=None, loss_weight=loss_weight)
        return ul_loss

    def cons_rampup_func(self) -> float:
        max_iter = self.cfg.SOLVER.MAX_ITER
        rampup_schedule = self.cfg.ALGORITHM.CONS_RAMPUP_SCHEDULE
        rampup_ratio = self.cfg.ALGORITHM.CONS_RAMPUP_ITERS_RATIO
        rampup_iter = max_iter * rampup_ratio

        if rampup_schedule == "linear":
            rampup_value = min(float(self.iter) / rampup_iter, 1.0)
        elif rampup_schedule == "exp":
            rampup_value = math.exp(-5 * (1 - min(float(self.iter) / rampup_iter, 1))**2)
        return rampup_value

    def train(self) -> None:
        super().train()

    def evaluate(self, model=None):
        if self.cfg.ALGORITHM.NAME == "cRT":
            eval_model = self.model
        else:
            eval_model = self.ema_model
        return super().evaluate(eval_model)
