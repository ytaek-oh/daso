import time

import torch
from yacs.config import CfgNode

from .base_ssl_algorithm import SemiSupervised


class USADTM(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        self.pretrain_steps = cfg.ALGORITHM.USADTM.PRETRAIN_STEPS
        self.warmup_cluster_loss = cfg.ALGORITHM.USADTM.WARMUP_CLUSTER_LOSS
        self.with_ema_proto = cfg.ALGORITHM.USADTM.WITH_EMA_PROTOTYPE

    def run_step(self) -> None:
        self.model.l_loss = self.l_loss
        if (self.iter + 1) >= self.warmup_cluster_loss:
            self.model.apply_uc_loss = True

        loss_dict = {}
        metrics_dict = {}

        # measure data time
        start = time.perf_counter()
        l_images, labels, _ = next(self._l_iter)
        (ul_identity, ul_weak, ul_strong), UL_LABELS, _ = next(self._ul_iter)
        data_time = time.perf_counter() - start

        # load images and labels onto gpu
        if torch.cuda.is_available():
            l_images = l_images.to(self.device)
            labels = labels.to(self.device).long()
            ul_identity = ul_identity.to(self.device)
            ul_weak = ul_weak.to(self.device)
            ul_strong = ul_strong.to(self.device)
            UL_LABELS = UL_LABELS.to(self.device)  # GT UNLABELED LABELS

        if self.with_ema_proto:
            raise NotImplementedError
        else:
            ema_model = self.model
        input_concat = torch.cat([l_images, ul_identity, ul_weak, ul_strong], 0)
        loss_dict = self.model(
            input_concat,
            is_train=True,
            labels=labels,
            ema_model=ema_model,
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
