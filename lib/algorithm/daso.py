import time

import torch
from yacs.config import CfgNode

from .base_ssl_algorithm import SemiSupervised


class DASO(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        self.pretrain_steps = cfg.ALGORITHM.DASO.PRETRAIN_STEPS
        self.dist_logger.accumulate_pl = True

        class_count = self.get_label_dist(device=self.device)
        self.model.target_dist = class_count / class_count.sum()  # probability
        self.model.bal_param = class_count[-1] / class_count  # bernoulli parameter

    def run_step(self) -> None:
        self.model.l_loss = self.l_loss
        self.model.iter = self.iter

        loss_dict = {}
        metrics_dict = {}

        # measure data time
        start = time.perf_counter()
        l_images, labels, _ = next(self._l_iter)
        (ul_weak, ul_strong), UL_LABELS, ul_indices = next(self._ul_iter)
        data_time = time.perf_counter() - start

        # load images and labels onto gpu
        if torch.cuda.is_available():
            l_images = l_images.to(self.device)
            labels = labels.to(self.device).long()
            ul_weak = ul_weak.to(self.device)
            ul_strong = ul_strong.to(self.device)
            UL_LABELS = UL_LABELS.to(self.device)

        # input concatenation
        input_concat = torch.cat([l_images, ul_weak, ul_strong], 0)
        loss_dict = self.model(
            input_concat,
            is_train=True,
            labels=labels,
            ema_model=self.ema_model,
            dist_logger=self.dist_logger,
            ul_loss=self.ul_loss,
            UL_LABELS=UL_LABELS
        )

        losses = sum(loss_dict.values())
        metrics_dict.update(loss_dict)

        # update params and schedule learning rates
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        if self.apply_scheduler:
            self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]["lr"]
        ema_decay = self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)

        # measure iter time
        iter_time = time.perf_counter() - start

        # logging
        self.iter_timer.update(iter_time, n=l_images.size(0))
        self.meters.put_scalar(
            "misc/iter_time", self.iter_timer.avg, n=l_images.size(0), show_avg=False
        )
        self.meters.put_scalar("train/ema_decay", ema_decay, show_avg=False)
        self.meters.put_scalar("misc/data_time", data_time, n=l_images.size(0))
        self.meters.put_scalar("misc/lr", current_lr, show_avg=False)

        # make a log for accuracy and losses
        self._write_metrics(metrics_dict, n=l_images.size(0), prefix="train")

        if self.iter + 1 >= self.pretrain_steps:
            self.model.pretraining = False

        # pl dist update period
        if (
            (self.iter + 1) % self.cfg.ALGORITHM.DASO.PL_DIST_UPDATE_PERIOD == 0
            and (self.iter + 1) < self.max_iter
        ):
            assert self.dist_logger.accumulate_pl
            self.dist_logger.update_pl_dist()
