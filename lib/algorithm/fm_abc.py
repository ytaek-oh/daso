import time

import torch
import torch.nn.functional as F
from yacs.config import CfgNode

from .fixmatch import FixMatch


class FixMatchABC(FixMatch):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        class_count = self.get_label_dist(device=self.device)

        # we believe the last element to be the most tail class.
        self.bal_param = class_count[-1] / class_count  # bernoulli parameter

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

        # feature extraction
        input_concat = torch.cat([l_images, ul_weak, ul_strong], 0)
        feats_concat = self.model(input_concat, return_features=True)

        # logits for ABC
        logits_concat_abc = self.model.abc_classify(feats_concat)
        l_logits_abc = logits_concat_abc[:num_labels]

        l_mask_abc = torch.bernoulli(self.bal_param[labels].detach()).float()
        cls_loss_abc = self.l_loss(l_logits_abc, labels, weight=l_mask_abc)

        # unlabeled data part
        logits_weak_abc, logits_strong_abc = logits_concat_abc[num_labels:].chunk(2)
        p_abc = logits_weak_abc.detach_().softmax(dim=1)  # soft pseudo labels
        with torch.no_grad():
            conf_abc, pred_class_abc = torch.max(p_abc, dim=1)
            loss_weight_abc = conf_abc.ge(self.conf_thres).float()

        # mask generation
        assert self.cfg.PERIODS.EVAL == 500
        current_epoch = int(self.iter / 500)  # 0~499
        gradual_bal_param = 1.0 - (current_epoch / 500) * (1.0 - self.bal_param)
        ul_mask_abc = torch.bernoulli(gradual_bal_param[pred_class_abc].detach()).float()

        # mask consistency loss with soft pseudo-label
        abc_mask = loss_weight_abc * ul_mask_abc
        cons_loss_abc = -1 * torch.mean(
            abc_mask * torch.sum(p_abc * F.log_softmax(logits_strong_abc, dim=1), dim=1)
        )

        abc_loss = cls_loss_abc + cons_loss_abc
        loss_dict.update({"loss_abc": abc_loss})

        # loss computation for SSL learner.
        logits_concat = self.model.classify(feats_concat)
        l_logits = logits_concat[:num_labels]
        cls_loss = self.l_loss(l_logits, labels)
        loss_dict.update({"loss_cls": cls_loss})

        # unlabeled loss
        logits_weak, logits_strong = logits_concat[num_labels:].chunk(2)
        p = logits_weak.detach_().softmax(dim=1)  # soft pseudo labels
        with torch.no_grad():
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
