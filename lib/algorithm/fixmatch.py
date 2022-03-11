import time

import torch
from yacs.config import CfgNode

from .base_ssl_algorithm import SemiSupervised


class FixMatch(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)

    def run_step(self) -> None:
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

        num_labels = labels.size(0)

        # input concatenation
        input_concat = torch.cat([l_images, ul_weak, ul_strong], 0)

        # predictions
        logits_concat = self.model(input_concat)

        # loss computation
        l_logits = logits_concat[:num_labels]

        # logit adjustment in train-time.
        if self.with_la:
            l_logits += (self.tau * self.p_data.view(1, -1).log())

        cls_loss = self.l_loss(l_logits, labels)
        loss_dict.update({"loss_cls": cls_loss})

        # unlabeled loss
        logits_weak, logits_strong = logits_concat[num_labels:].chunk(2)
        p = logits_weak.detach_().softmax(dim=1)  # soft pseudo labels
        if self.with_align:
            p = self.dist_align(p)  # distribution alignment

        with torch.no_grad():
            if self.with_darp:        
                p = self.darp_optimizer.step(p, ul_indices)
            # final pseudo-labels with confidence
            confidence, pred_class = torch.max(p, dim=1)

        loss_weight = confidence.ge(self.conf_thres).float()
        cons_loss = self.ul_loss(
            logits_strong, pred_class, weight=loss_weight, avg_factor=ul_weak.size(0)
        )
        loss_dict.update({"loss_cons": cons_loss})
        losses = sum(loss_dict.values())

        # compute batch-wise accuracy and update metrics_dict
        top1, top5 = self.accuracy(l_logits, labels)
        metrics_dict.update(loss_dict)
        metrics_dict.update({"top1": top1, "top5": top5})

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
