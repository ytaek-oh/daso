import os
import time
from collections import OrderedDict

import torch
from yacs.config import CfgNode

from .base_ssl_algorithm import SemiSupervised


class cRT(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        self.rebuild_model(cfg)

    def rebuild_model(self, cfg):
        target_dir = cfg.ALGORITHM.CRT.TARGET_DIR
        assert target_dir
        state_dict = torch.load(os.path.join(target_dir, "model_final.pth.tar"))
        self.logger.info(f"loaded the final checkpoint from the path: {target_dir}.")
        self.logger.info(state_dict["meta"])

        # load checkpoint for "model"
        self.logger.info("loading model state dict, only feature encoder")
        self.load_model_checkpoint(self.model, state_dict["model"])
        if state_dict["ema_model"] is not None:
            # load checkpoint for "ema_model"
            self.logger.info("loading ema model state dict...")
            self.load_model_checkpoint(
                self.ema_model.ema_model, state_dict["ema_model"], load_classifier=True
            )
        else:
            self.logger.info(
                "ema_model is not detected on the target checkpoint. "
                "Just copying the original network."
            )
        test_top1, ema_test_top1 = self.evaluate(self.model)
        self.logger.info(f"test_top1: {test_top1}, ema_test_top1: {ema_test_top1}")

    def load_model_checkpoint(self, model, model_state_dict, load_classifier=False):
        encoder_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k.startswith("ema_model."):
                k = k.replace("ema_model.", "")
            if k.startswith("encoder."):
                encoder_state_dict[k[8:]] = v
        model.encoder.load_state_dict(encoder_state_dict)
        for p in model.encoder.parameters():
            p.requires_grad_(False)

        if load_classifier:
            classifier_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                if k.startswith("ema_model."):
                    k = k.replace("ema_model.", "")
                if k.startswith("classifier."):
                    classifier_state_dict[k[11:]] = v
            model.classifier.load_state_dict(classifier_state_dict)

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
        ema_decay = self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        self.meters.put_scalar("train/ema_decay", ema_decay, show_avg=False)
        self.meters.put_scalars(
            {
                "data_time": data_time,
                "lr": current_lr
            }, show_avg=False, prefix="misc"
        )
        self.meters.put_scalar("misc/iter_time", iter_time, n=l_images.size(0))

        # make a log for accuracy and losses
        self._write_metrics(metrics_dict, n=l_images.size(0), prefix="train")

    def evaluate(self, model) -> float:
        return super().evaluate()
