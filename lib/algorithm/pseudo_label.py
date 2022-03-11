import time

import torch
from yacs.config import CfgNode

from .base_ssl_algorithm import SemiSupervised


class PseudoLabel(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)

    def run_step(self) -> None:
        loss_dict = {}
        metrics_dict = {}

        # measure data time
        start = time.perf_counter()
        l_images, labels, _ = next(self._l_iter)
        ul_images, UL_LABELS, _ = next(self._ul_iter)
        data_time = time.perf_counter() - start

        # load images and labels onto gpu
        if torch.cuda.is_available():
            l_images = l_images.to(self.device)
            labels = labels.to(self.device).long()
            ul_images = ul_images.to(self.device)
            UL_LABELS = UL_LABELS.to(self.device)

        # input concatenation
        input_concat = torch.cat([l_images, ul_images], 0)

        # predictions
        logits_concat = self.model(input_concat)

        # loss computation
        cls_loss = self.l_loss(logits_concat[:len(labels)], labels)
        loss_dict.update({"loss_cls": cls_loss})

        # unlabeled loss
        ul_prob = logits_concat[len(labels):].detach().softmax(dim=1)
        confidence, pred_class = torch.max(ul_prob, dim=1)

        weight = confidence.ge(self.conf_thres).float()

        cons_loss = self.ul_loss(
            logits_concat[len(labels):], pred_class, weight=weight, avg_factor=ul_images.size(0)
        )
        loss_dict.update({"loss_cons": self.cons_rampup_func() * cons_loss})
        losses = sum(loss_dict.values())

        # compute batch-wise accuracy and update metrics_dict
        top1, top5 = self.accuracy(logits_concat[:len(labels)], labels)
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
        self.meters.put_scalar("train/cons_rampup_coeff", self.cons_rampup_func(), show_avg=False)
        self.meters.put_scalar("train/ema_decay", ema_decay, show_avg=False)
        self.meters.put_scalar("misc/data_time", data_time, n=l_images.size(0))
        self.meters.put_scalar("misc/lr", current_lr, show_avg=False)

        # make a log for accuracy and losses
        self._write_metrics(metrics_dict, n=l_images.size(0), prefix="train")
