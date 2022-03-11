import time

import torch
import torch.nn as nn
from yacs.config import CfgNode

from lib.models.feature_queue import FeatureQueue

from .base_ssl_algorithm import SemiSupervised


class MeanTeacher(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        # self.with_scl = cfg.MODEL.LOSS.WITH_SUPPRESSED_CONSISTENCY

        # DASO options
        self.with_daso = cfg.ALGORITHM.MEANTEACHER.APPLY_DASO
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

    def run_step(self) -> None:
        loss_dict = {}
        metrics_dict = {}

        # measure data time
        start = time.perf_counter()
        l_images, labels, _ = next(self._l_iter)
        (ul_student, ul_teacher), UL_LABELS, _ = next(self._ul_iter)
        data_time = time.perf_counter() - start

        # load images and labels onto gpu
        if torch.cuda.is_available():
            l_images = l_images.to(self.device)
            labels = labels.to(self.device).long()
            ul_student = ul_student.to(self.device)
            ul_teacher = ul_teacher.to(self.device)
        num_labels = labels.size(0)

        # input concatenation
        input_student = torch.cat([l_images, ul_student], 0)
        if self.with_daso:
            input_teacher = torch.cat([l_images, ul_teacher])
        else:
            input_teacher = ul_teacher

        # predictions
        if self.with_daso:
            # push labeled data into memory queue
            with torch.no_grad():
                _feats_teacher = self.ema_model(input_teacher, return_features=True)
                l_feats, feats_teacher = _feats_teacher.chunk(2)
                self.queue.enqueue(l_feats.clone().detach(), labels.clone().detach())

            feats_student = self.model.encoder(input_student)  # (l, ul)

            # representation-level learning.
            assignment = torch.Tensor([-1 for _ in range(len(UL_LABELS))]).float().to(self.device)
            if not self.pretraining:
                prototypes = self.queue.prototypes  # (K, D)

                with torch.no_grad():
                    # similarity between weak features and prototypes  (B, K)
                    sim_weak = self.similarity_fn(
                        feats_teacher.unsqueeze(1), prototypes.unsqueeze(0)
                    ) / self.T_proto
                    soft_target = sim_weak.softmax(dim=1)
                    _, assignment = torch.max(soft_target.detach(), dim=1)

            pred_student = self.model.classifier(feats_student)  # (l, ul)
            with torch.no_grad():
                pred_teacher = self.ema_model.ema_model.classifier(feats_teacher)

        else:
            pred_student = self.model(input_student)
            pred_teacher = self.ema_model(input_teacher)

        # loss computation
        cls_loss = self.l_loss(pred_student[:num_labels], labels)
        loss_dict.update({"loss_cls": cls_loss})

        # pred_teacher -> linear pl by ema
        with torch.no_grad():
            confidence, pred_class = torch.max(pred_teacher.softmax(dim=1), dim=1)

        if self.with_daso:
            with torch.no_grad():
                p = pred_teacher.softmax(dim=1)
                if (not self.pretraining):
                    # Generate DASO pseudo-label
                    current_pl_dist = self.dist_logger.get_pl_dist().to(self.device)
                    current_pl_dist = current_pl_dist**(1. / self.T_dist)
                    current_pl_dist = current_pl_dist / current_pl_dist.sum()
                    current_pl_dist = current_pl_dist / current_pl_dist.max()

                    pred_to_dist = current_pl_dist[pred_class].view(-1, 1)
                    if not self.with_dist_aware:
                        pred_to_dist = self.interp_alpha

                    # p = pred_teacher.softmax(dim=1)
                    # pl mixup
                    p = (1. - pred_to_dist) * p + pred_to_dist * soft_target
                    confidence, pred_class = torch.max(p.detach(), dim=1)  # modifided pseudo-labels
                self.dist_logger.push_pl_list(pred_class)

            # loss
            cons_loss = self.ul_loss(pred_student[num_labels:].softmax(1), p.detach_())
        else:
            # normal mean teacher
            p = pred_teacher.softmax(dim=1)
            confidence, pred_class = torch.max(p.detach(), dim=1)  # linear pseudo-labels
            cons_loss = self.ul_loss(
                pred_student[num_labels:].softmax(1),
                pred_teacher.softmax(1).detach_()
            )

        loss_dict.update({"loss_cons": self.cons_rampup_func() * cons_loss})
        losses = sum(loss_dict.values())

        # compute batch-wise accuracy and update metrics_dict
        top1, top5 = self.accuracy(pred_student[:len(labels)], labels)
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
        self.meters.put_scalars(
            {
                "data_time": data_time,
                "lr": current_lr
            }, show_avg=False, prefix="misc"
        )
        self.meters.put_scalar("train/cons_rampup_coeff", self.cons_rampup_func(), show_avg=False)
        self.meters.put_scalar("train/ema_decay", ema_decay, show_avg=False)

        self.iter_timer.update(iter_time, n=l_images.size(0))
        self.meters.put_scalar(
            "misc/iter_time", self.iter_timer.avg, n=l_images.size(0), show_avg=False
        )

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
