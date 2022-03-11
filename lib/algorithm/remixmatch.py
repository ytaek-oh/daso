import time

import numpy as np
import torch
import torch.nn.functional as F
from yacs.config import CfgNode

from typing import NoReturn

from .base_ssl_algorithm import SemiSupervised
from .darp_reproduce import get_target_dist
from .ssl_utils import interleave


class ReMixMatch(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        # mixmatch parameters
        self.K = cfg.ALGORITHM.REMIXMATCH.NUM_AUG
        self.T = cfg.ALGORITHM.REMIXMATCH.TEMPERATURE
        self.alpha = cfg.ALGORITHM.REMIXMATCH.MIXUP_ALPHA

        # loss weights
        self.weight_kl = cfg.ALGORITHM.REMIXMATCH.WEIGHT_KL
        self.weight_rot = cfg.ALGORITHM.REMIXMATCH.WEIGHT_ROT

        # override DARP's target distribution
        if self.with_align and self.with_darp:
            target_dist = get_target_dist(cfg, to_prob=True, device=self.device)
            self.dist_align.set_target_dist(target_dist)

    def run_step(self) -> NoReturn:
        loss_dict = {}
        metrics_dict = {}

        # measure data time
        start = time.perf_counter()
        l_images, labels, _ = next(self._l_iter)
        ul_images, UL_LABELS, ul_indices = next(self._ul_iter)  # (weak, strong1, strong2)
        ul_images = torch.cat(ul_images, 0)
        data_time = time.perf_counter() - start
        batch_size = labels.size(0)

        # load images and labels onto gpu
        if torch.cuda.is_available():
            l_images = l_images.to(self.device)
            labels = labels.to(self.device).long()
            ul_images = ul_images.to(self.device)
            UL_LABELS = UL_LABELS.to(self.device)

        ul_weak, ul_strong1, ul_strong2 = ul_images.chunk(3)

        # guess label; DA (ReMixMatch) -> Temperature -> DARP
        with torch.no_grad():
            ul_pred = self.model(ul_weak)
            p = ul_pred.softmax(dim=1)  # predicted soft pseudo label
            if self.with_align:
                p = self.dist_align(p)

            # temperature scaling
            pt = p**(1. / self.T)
            ul_targets = pt / pt.sum(dim=1, keepdim=True).detach()

            # DARP
            if self.with_darp:
                with torch.no_grad():
                    ul_targets = self.darp_optimizer.step(ul_targets, ul_indices)
            confidence, pred_class = torch.max(ul_targets, dim=1)

        # mixup
        bin_labels = F.one_hot(labels, num_classes=self.num_classes).float()
        all_images = torch.cat([l_images, ul_weak, ul_strong1, ul_strong2], dim=0)
        all_labels = torch.cat([bin_labels, ul_targets, ul_targets, ul_targets], dim=0)

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

        # put interleaved samples back
        logits = interleave(logits, batch_size)

        l_logits = logits[0]
        ul_logits = torch.cat(logits[1:], dim=0)

        # classification loss
        loss_cls = -1 * torch.sum(
            mixed_labels[:len(labels)] * F.log_softmax(l_logits, dim=1), dim=1
        ).mean()
        loss_dict.update({"loss_cls": loss_cls})

        # unlabeled loss
        loss_cons = -1 * self.ul_loss.loss_weight * torch.sum(
            mixed_labels[len(labels):] * F.log_softmax(ul_logits, dim=1), dim=1
        ).mean()
        loss_dict.update({"loss_cons": self.cons_rampup_func() * loss_cons})

        # premixup loss
        logits_strong1 = self.model(ul_strong1)
        loss_kl = -1 * self.weight_kl * torch.sum(
            ul_targets * F.log_softmax(logits_strong1, dim=1), dim=1
        ).mean()
        loss_dict.update({"loss_kl": self.cons_rampup_func() * loss_kl})

        # rotation branch
        rot_images = []
        rot_labels = torch.randint(0, 4, (ul_strong1.size(0), )).long().to(self.device)
        for i in range(ul_strong1.size(0)):
            inputs_rot = torch.rot90(ul_strong1[i], rot_labels[i], [1, 2]).reshape(1, 3, 32, 32)
            rot_images.append(inputs_rot)
        rot_images = torch.cat(rot_images, 0).to(self.device)
        rot_logits = self.model(rot_images, classification_mode="rotation")
        rot_loss = self.weight_rot * F.cross_entropy(rot_logits, rot_labels,
                                                     reduction="none").sum() / rot_labels.size(0)
        loss_dict.update({"loss_rotation": rot_loss})
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
