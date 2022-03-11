import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode

from lib.models.feature_queue import FeatureQueue
from typing import NoReturn

from .base_ssl_algorithm import SemiSupervised
from .ssl_utils import interleave


class MixMatch(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        self.K = cfg.ALGORITHM.MIXMATCH.NUM_AUG
        assert self.K == 2

        self.T = cfg.ALGORITHM.MIXMATCH.TEMPERATURE
        self.alpha = cfg.ALGORITHM.MIXMATCH.MIXUP_ALPHA

        # DASO options
        self.with_daso = cfg.ALGORITHM.MIXMATCH.APPLY_DASO
        if self.with_daso:
            self.pretrain_steps = cfg.ALGORITHM.DASO.PRETRAIN_STEPS
            self.dist_logger.accumulate_pl = True
            self.queue = FeatureQueue(cfg)
            self.similarity_fn = nn.CosineSimilarity(dim=2)
            self.T_proto = cfg.ALGORITHM.DASO.PROTO_TEMP
            self.pretraining = True
            self.psa_loss_weight = cfg.ALGORITHM.DASO.PSA_LOSS_WEIGHT
            self.T_dist = cfg.ALGORITHM.DASO.DIST_TEMP
            self.with_dist_aware = cfg.ALGORITHM.DASO.WITH_DIST_AWARE
            self.interp_alpha = cfg.ALGORITHM.DASO.INTERP_ALPHA

    def run_step(self) -> NoReturn:
        loss_dict = {}
        metrics_dict = {}

        # measure data time
        start = time.perf_counter()
        l_images, labels, _ = next(self._l_iter)
        (ul_images1, ul_images2), UL_LABELS, ul_indices = next(self._ul_iter)
        data_time = time.perf_counter() - start
        batch_size = labels.size(0)

        # load images and labels onto gpu
        if torch.cuda.is_available():
            l_images = l_images.to(self.device)
            labels = labels.to(self.device).long()
            ul_images1 = ul_images1.to(self.device)  # (B*K, CHW)
            ul_images2 = ul_images2.to(self.device)
            UL_LABELS = UL_LABELS.to(self.device)

        # predictions
        if self.with_daso:
            # push labeled data into memory queue
            with torch.no_grad():
                l_feats = self.ema_model(l_images, return_features=True)
                self.queue.enqueue(l_feats.clone().detach(), labels.clone().detach())

        # guess label
        assignment = torch.Tensor([-1 for _ in range(len(UL_LABELS))]).float().to(self.device)
        with torch.no_grad():
            if self.with_daso and (not self.pretraining):
                ul_feat1 = self.model(ul_images1, return_features=True)
                prototypes = self.queue.prototypes  # (K, D)

                # similarity between weak features and prototypes  (B, K)
                sim_weak = self.similarity_fn(
                    ul_feat1.unsqueeze(1), prototypes.unsqueeze(0)
                ) / self.T_proto
                soft_target = sim_weak.softmax(dim=1)
                assign_confidence, assignment = torch.max(soft_target.detach(), dim=1)

                ul_pred1 = self.model.classifier(ul_feat1)
            else:
                ul_pred1 = self.model(ul_images1)

            ul_pred2 = self.model(ul_images2)
            p = (ul_pred1.softmax(dim=1) + ul_pred2.softmax(dim=1)) / 2
            pt = p**(1. / self.T)

            # linear pseudo-label
            ul_targets = pt / pt.sum(dim=1, keepdim=True).detach()
            confidence, pred_class = torch.max(ul_targets, dim=1)

            if self.with_daso and (not self.pretraining):
                current_pl_dist = self.dist_logger.get_pl_dist().to(self.device)
                current_pl_dist = current_pl_dist**(1. / self.T_dist)
                current_pl_dist = current_pl_dist / current_pl_dist.sum()
                current_pl_dist = current_pl_dist / current_pl_dist.max()

                pred_to_dist = current_pl_dist[pred_class].view(-1, 1)
                if not self.with_dist_aware:
                    pred_to_dist = self.interp_alpha

                # pl mixup
                ul_targets = (1. - pred_to_dist) * ul_targets + pred_to_dist * soft_target
            confidence, pred_class = torch.max(ul_targets.detach(), dim=1)  # NEW PL

            if self.with_daso:
                self.dist_logger.push_pl_list(pred_class)

        # mixup
        bin_labels = F.one_hot(labels, num_classes=ul_targets.size(1)).float()
        all_images = torch.cat([l_images, ul_images1, ul_images2], dim=0)
        all_labels = torch.cat([bin_labels, ul_targets, ul_targets], dim=0)

        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1.0 - lam)

        rand_idx = torch.randperm(all_images.size(0))
        mixed_images = lam * all_images + (1.0 - lam) * all_images[rand_idx]
        mixed_labels = lam * all_labels + (1.0 - lam) * all_labels[rand_idx]

        # predictions
        mixed_images = list(torch.split(mixed_images, batch_size))
        mixed_images = interleave(mixed_images, batch_size)
        logits = [self.model(mixed_images[0])]
        for images in mixed_images[1:]:
            logits.append(self.model(images))
        logits = interleave(logits, batch_size)

        l_logits = logits[0]
        ul_logits = torch.cat(logits[1:], dim=0)

        # loss computation
        cls_loss = -(F.log_softmax(l_logits, dim=1) * mixed_labels[:len(labels)]).sum(dim=1).mean()
        loss_dict.update({"loss_cls": cls_loss})

        cons_loss = self.ul_loss(ul_logits.softmax(dim=1), mixed_labels[len(labels):])
        # cons_loss = torch.mean((ul_logits.softmax(dim=1) - mixed_labels[len(labels):])**2)
        loss_dict.update({"loss_cons": self.cons_rampup_func() * cons_loss})
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
        self.meters.put_scalar("train/cons_rampup_coeff", self.cons_rampup_func(), show_avg=False)
        self.meters.put_scalar("train/ema_decay", ema_decay, show_avg=False)

        self.iter_timer.update(iter_time, n=l_images.size(0))
        self.meters.put_scalar(
            "misc/iter_time", self.iter_timer.avg, n=l_images.size(0), show_avg=False
        )
        self.meters.put_scalar("misc/data_time", data_time, n=l_images.size(0))
        self.meters.put_scalar("misc/lr", current_lr, show_avg=False)

        # make a log for accuracy and losses
        self._write_metrics(metrics_dict, n=l_images.size(0), prefix="train")
        if self.with_daso:
            if self.iter + 1 >= self.pretrain_steps:
                self.pretraining = False

            # pl dist update period
            if (
                (self.iter + 1) % self.cfg.ALGORITHM.DASO.PL_DIST_UPDATE_PERIOD == 0
                and (self.iter + 1) < self.max_iter
            ):
                assert self.dist_logger.accumulate_pl
                self.dist_logger.update_pl_dist()
