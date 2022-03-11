import time

import torch

from .base_algorithm import BaseAlgorithm


class Supervised(BaseAlgorithm):

    def __init__(self, cfg):
        super().__init__(cfg)

    def run_step(self):
        loss_dict = {}
        metrics_dict = {}

        # measure data time
        start = time.perf_counter()
        l_images, labels, _ = next(self._l_iter)
        data_time = time.perf_counter() - start

        # load images and labels onto gpu
        if torch.cuda.is_available():
            l_images = l_images.to(self.device)
            labels = labels.to(self.device).long()
        l_outputs = self.model(l_images)

        # logit adjustment
        if self.with_la:
            l_outputs += (self.tau * self.p_data.view(1, -1).log())

        # compute loss
        cls_loss = self.l_loss(l_outputs, labels)
        loss_dict.update({"loss_cls": cls_loss})
        losses = sum(loss_dict.values())

        # compute batch-wise accuracy and update metrics_dict
        top1, top5 = self.accuracy(l_outputs, labels)
        metrics_dict.update(loss_dict)
        metrics_dict.update({"top1": top1, "top5": top5})

        # update params and schedule learning rates
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        if self.apply_scheduler:
            self.scheduler.step()

        # measure iter time
        iter_time = time.perf_counter() - start

        # logging
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.meters.put_scalars(
            {
                "data_time": data_time,
                "lr": current_lr
            }, show_avg=False, prefix="misc"
        )
        self.meters.put_scalar("misc/iter_time", iter_time, n=l_images.size(0))

        # make a log for accuracy and losses
        self._write_metrics(metrics_dict, n=l_images.size(0), prefix="train")
