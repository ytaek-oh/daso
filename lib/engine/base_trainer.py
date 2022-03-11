import logging
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from lib.dataset import build_data_loaders
from lib.dataset.utils import get_data_config
from lib.models import build_model
from lib.models.losses import Accuracy, build_loss
from lib.solver import build_lr_scheduler, build_optimizer
from lib.utils import AverageMeter, Meters, get_last_n_median, save_checkpoint
from lib.utils.writer import CommonMetricPrinter, JSONWriter, TensorboardWriter
from typing import Tuple


class BaseTrainer:
    """class for BaseTrainer"""

    def __init__(
        self, cfg, model, optimizer, l_loader, ul_loader=None, valid_loader=None, test_loader=None
    ):
        # configuration
        model.train()
        self.cfg = cfg
        self.data_cfg = get_data_config(cfg)
        self.device = cfg.GPU_ID

        # data loaders
        self.l_loader = l_loader
        self._l_iter = iter(l_loader)

        self.with_ul = ul_loader is not None
        if self.with_ul:
            self.ul_loader = ul_loader
            self._ul_iter = iter(ul_loader)

        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # build model and losses
        self.model = model
        self.accuracy = Accuracy(model.num_classes)
        self.l_loss = self.build_labeled_loss(cfg)

        # optimizer
        self.optimizer = optimizer

        # scheduler
        self.apply_scheduler = cfg.SOLVER.APPLY_SCHEDULER
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # training steps
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER

        # for logging purpose
        self.logger = logging.getLogger()
        self.meters = Meters()
        self.writers = self._build_writers(cfg)
        self.iter_timer = AverageMeter()
        self.eval_history = defaultdict(list)

    def load_checkpoint(self, resume: str) -> None:
        self.logger.info(f"resume checkpoint from: {resume}")

        state_dict = torch.load(resume)
        # load model
        self.model.load_state_dict(state_dict["model"])

        # load ema model
        if self.with_ul and state_dict["ema_model"] is not None:
            self.ema_model.load_state_dict(state_dict["ema_model"])

        # load optimizer and scheduler
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])

        # process meta information
        start_iter = 0
        meta_dict = state_dict["meta"]
        if meta_dict is not None and "iter" in meta_dict.keys():
            start_iter = meta_dict["iter"] + 1

        dict_str = "  ".join([f"{k}: {v}" for k, v in meta_dict.items() if "iter" not in k])
        self.logger.info(
            "Successfully loaded the checkpoint. "
            f"start_iter: {start_iter}  "
            f"intermediate status: {dict_str}"
        )

        # loaded
        self.start_iter = start_iter

    def save_checkpoint(self, *, save_ema_model: bool = False) -> None:
        # meta information construction
        meta_dict = {"iter": self.iter + 1}
        for prefix, history in self.eval_history.items():
            current_val = history[-1]
            max_val = max(history)
            meta_dict[prefix] = current_val
            meta_dict[prefix + "_" + "best"] = max_val
            meta_dict[prefix + "_" + "median20"] = get_last_n_median(history, n=20)

        is_best = False
        prefix = "valid/top1"
        if meta_dict[prefix] >= meta_dict[prefix + "_best"]:
            is_best = True

        is_final_iter = self.iter + 1 == self.max_iter
        checkpoint_name = "model_final.pth.tar" if is_final_iter else "checkpoint.pth.tar"
        save_checkpoint(
            self.cfg.OUTPUT_DIR,
            self.model,
            self.optimizer,
            self.scheduler,
            is_best=is_best,
            ema_model=self.ema_model if self.with_ul else None,
            meta_dict=meta_dict,
            file_name=checkpoint_name
        )

    def _build_writers(self, cfg: CfgNode) -> list:
        writers = (
            [
                CommonMetricPrinter(max_iter=self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardWriter(log_dir=self.cfg.OUTPUT_DIR)
            ]
        )
        return writers

    def build_labeled_loss(self, cfg: CfgNode, warmed_up=False) -> nn.Module:
        loss_type = cfg.MODEL.LOSS.LABELED_LOSS
        num_classes = cfg.MODEL.NUM_CLASSES
        assert loss_type == "CrossEntropyLoss"

        class_count = self.get_label_dist(device=self.device)
        per_class_weights = None
        if cfg.MODEL.LOSS.WITH_LABELED_COST_SENSITIVE and warmed_up:
            loss_override = cfg.MODEL.LOSS.COST_SENSITIVE.LOSS_OVERRIDE
            beta = cfg.MODEL.LOSS.COST_SENSITIVE.BETA
            if beta < 1:
                # effective number of samples;
                effective_num = 1.0 - torch.pow(beta, class_count)
                per_class_weights = (1.0 - beta) / effective_num
            else:
                per_class_weights = 1.0 / class_count

            # sum to num_classes
            per_class_weights = per_class_weights / torch.sum(per_class_weights) * num_classes

            if loss_override == "":
                # CE loss
                loss_fn = build_loss(
                    cfg, loss_type, class_count=class_count, class_weight=per_class_weights
                )

            elif loss_override == "LDAM":
                # LDAM loss
                loss_fn = build_loss(
                    cfg, "LDAMLoss", class_count=class_count, class_weight=per_class_weights
                )

            else:
                raise ValueError()
        else:
            loss_fn = build_loss(
                cfg, loss_type, class_count=class_count, class_weight=per_class_weights
            )

        return loss_fn

    @classmethod
    def build_model(cls, cfg: CfgNode) -> nn.Module:
        model = build_model(cfg)
        return model

    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model: nn.Module) -> optim.Optimizer:
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(
        cls, cfg: CfgNode, optimizer: optim.Optimizer, override_max_iter=None
    ) -> optim.lr_scheduler._LRScheduler:
        return build_lr_scheduler(cfg, optimizer, override_max_iter=override_max_iter)

    @classmethod
    def build_data_loaders(cls, cfg: CfgNode) -> Tuple[DataLoader]:
        return build_data_loaders(cfg)
