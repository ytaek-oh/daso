import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode

from lib.models.feature_queue import FeatureQueue

from .remixmatch import ReMixMatch
from .ssl_utils import interleave


class ReMixMatchDASO(ReMixMatch):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        self.pretrain_steps = cfg.ALGORITHM.DASO.PRETRAIN_STEPS
        self.dist_logger.accumulate_pl = True

        self.similarity_fn = nn.CosineSimilarity(dim=2)
        self.T_proto = cfg.ALGORITHM.DASO.PROTO_TEMP
        self.pretraining = True
        self.psa_loss_weight = cfg.ALGORITHM.DASO.PSA_LOSS_WEIGHT
        self.T_dist = cfg.ALGORITHM.DASO.DIST_TEMP
        self.queue = FeatureQueue(cfg)

    def run_step(self):
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

        logger_dict = {"gt_labels": labels, "ul_labels": UL_LABELS}  # initial log

        ul_weak, ul_strong1, ul_strong2 = ul_images.chunk(3)

        # push memory queue
        with torch.no_grad():
            l_feats = self.ema_model(l_images, return_features=True)
            self.queue.enqueue(l_feats.clone().detach(), labels.clone().detach())

        # feature space. [u_weak, u_strong1]
        ul_x = torch.cat([ul_weak, ul_strong1], 0)
        ul_feats = self.model.encoder(ul_x)
        feats_weak, feats_strong = ul_feats.chunk(2)  # (B, D)
        assignment = torch.Tensor([-1 for _ in range(len(UL_LABELS))]).float().to(self.device)
        if not self.pretraining:
            prototypes = self.queue.prototypes  # (K, D)

            with torch.no_grad():
                # similarity between weak features and prototypes  (B, K)
                sim_weak = self.similarity_fn(
                    feats_weak.unsqueeze(1), prototypes.unsqueeze(0)
                ) / self.T_proto
                soft_target = sim_weak.softmax(dim=1)
                assign_confidence, assignment = torch.max(soft_target.detach(), dim=1)
                logger_dict.update({"sem_pl": assignment})  # semantic pl

            # soft loss
            if self.psa_loss_weight > 0:
                # similarity between strong features and prototypes  (B, K)
                sim_strong = self.similarity_fn(
                    feats_strong.unsqueeze(1), prototypes.unsqueeze(0)
                ) / self.T_proto

                loss_assign = -1 * torch.sum(soft_target * F.log_softmax(sim_strong, dim=1),
                                             dim=1).sum() / sim_weak.size(0)
                loss_dict.update({"loss_assign": self.psa_loss_weight * loss_assign})

        # guess label; DA (ReMixMatch) -> Temperature -> DARP
        with torch.no_grad():
            ul_pred = self.model.classifier(feats_weak)
            p = ul_pred.softmax(dim=1)  # predicted soft pseudo label
            if self.with_align:
                p = self.dist_align(p)

            # temperature scaling
            pt = p**(1. / self.T)
            ul_targets = pt / pt.sum(dim=1, keepdim=True).detach()
            confidence, pred_class = torch.max(ul_targets, dim=1)

            # Apply DASO pseudo-label
            if not self.pretraining:
                current_pl_dist = self.dist_logger.get_pl_dist().to(self.device)  # (1, C)
                current_pl_dist = current_pl_dist**(1. / self.T_dist)
                current_pl_dist = current_pl_dist / current_pl_dist.sum()
                current_pl_dist = current_pl_dist / current_pl_dist.max()  # MIXUP

                pred_to_dist = current_pl_dist[pred_class].view(-1, 1)  # (B, )

                # pl mixup
                ul_targets = (1. - pred_to_dist) * ul_targets + pred_to_dist * soft_target
                confidence, pred_class = torch.max(ul_targets.detach(), dim=1)  # final pl
            self.dist_logger.push_pl_list(pred_class)
            logger_dict.update({"pseudo_labels": pred_class, "pl_confidence": confidence})

            # log distributions
            self.dist_logger.accumulate(logger_dict)

        # premixup loss
        logits_strong1 = self.model.classifier(feats_strong)
        loss_kl = -1 * self.weight_kl * torch.sum(
            ul_targets * F.log_softmax(logits_strong1, dim=1), dim=1
        ).mean()
        loss_dict.update({"loss_kl": self.cons_rampup_func() * loss_kl})

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

        if self.iter + 1 >= self.pretrain_steps:
            self.pretraining = False

        # pl dist update period
        if (
            (self.iter + 1) % self.cfg.ALGORITHM.DASO.PL_DIST_UPDATE_PERIOD == 0
            and (self.iter + 1) < self.max_iter
        ):
            assert self.dist_logger.accumulate_pl
            self.dist_logger.update_pl_dist()
