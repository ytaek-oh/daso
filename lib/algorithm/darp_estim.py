import os
import time

import numpy as np
import torch
from numpy.linalg import cond, inv
from yacs.config import CfgNode

from lib.dataset.utils import get_imb_num
from lib.utils import get_last_n_median

from .base_ssl_algorithm import SemiSupervised


def confusion(model, loader, num_class, device, is_unlabeled=False):
    model.eval()

    num_classes = torch.zeros(num_class)
    confusion = torch.zeros(num_class, num_class)

    for batch_idx, (inputs, targets, _) in enumerate(loader):
        batch_size = inputs.size(0)
        inputs, targets = inputs.to(device), targets.to(device).long()
        outputs = model(inputs)
        probs = torch.softmax(outputs.data, dim=1)

        # Update the confusion matrix
        for i in range(batch_size):
            target = targets[i]
            if target == -1:  # handling STL-10 dataset
                assert is_unlabeled
                target = 0
            confusion[:, target] += probs[i].cpu()
            num_classes[target] += 1

    if is_unlabeled:
        q_y_tilde = confusion.sum(1)  # gt label effect vanishes
        return q_y_tilde

    return confusion


def estimate_q_y(val_loader, u_loader, model, num_class, device):
    model.eval()

    conf_val = confusion(model, val_loader, num_class, device)  # est
    q_y_tilde = confusion(model, u_loader, num_class, device, is_unlabeled=True)  # pred

    for i in range(num_class):
        conf_val[:, i] /= conf_val[:, i].sum()

    cond_val = cond(conf_val.numpy())
    print(f"Condition value: {cond_val}")

    inv_conf_val = torch.Tensor(inv(conf_val.numpy()))
    q_y_esti = torch.matmul(inv_conf_val, q_y_tilde)

    return q_y_esti, cond_val


class DARP_ESTIM(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        self.best_acc = 0.

        # hyperparameters for estim.
        self.th_cond = cfg.ALGORITHM.DARP_ESTIM.THRESH_COND

        num_ul_head = self.data_cfg.NUM_UNLABELED_HEAD
        imb_factor_ul = self.data_cfg.IMB_FACTOR_UL
        reverse_ul = cfg.DATASET.REVERSE_UL_DISTRIBUTION
        self.ul_samples_per_class = torch.Tensor(
            get_imb_num(
                num_ul_head,
                imb_factor_ul,
                num_classes=self.num_classes,
                reverse=reverse_ul,
                normalize=False
            )
        ).float() if imb_factor_ul > 0 else None

        self.n_infer = 0
        self.final_q = torch.zeros(self.num_classes)

    def run_step(self) -> None:
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

        # predictions
        l_logits = self.model(l_images)

        # loss computation
        cls_loss = self.l_loss(l_logits, labels)
        loss_dict.update({"loss_cls": cls_loss})
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

    def train(self):
        super().train()
        # after training, ...
        final_q = (self.final_q / self.n_infer)
        print(final_q)
        np.save(os.path.join(self.cfg.OUTPUT_DIR, "estim.npy"), np.array(final_q))
        self.logger.info(
            "estimated distributions are saved to: {}".format(
                os.path.join(self.cfg.OUTPUT_DIR, "estim.npy")
            )
        )

    def evaluate(self, model, prefix=""):
        # evaluate via valid set and test set
        val_top1 = None
        if self.valid_loader:
            valid_results = self.eval_loop(self.model, self.valid_loader, prefix="valid")
            val_top1 = valid_results["top1"]
        if self.test_loader and self.cfg.EVAL_ON_TEST_SET:
            _ = self.eval_loop(self.model, self.test_loader, prefix="test")

        # calculate last 20 median
        metrics = {}
        prefixes = ["valid/top1", "test/top1"]
        for prefix in prefixes:
            if prefix in self.eval_history.keys():
                metric_key = prefix + "_median20"
                metrics[metric_key] = get_last_n_median(self.eval_history[prefix], n=20)

                # add to eval_history
                self.eval_history[metric_key].append(
                    get_last_n_median(self.eval_history[prefix], n=20)
                )
        if len(metrics.keys()) > 0:
            self.meters.put_scalars(metrics)

        # estimate confusion matrices
        est_q, cond_val = estimate_q_y(
            self.valid_loader, self.ul_loader, self.model, self.num_classes, self.device
        )

        is_zero = (est_q < 0).float().sum()
        is_best = (val_top1 > self.best_acc) and (is_zero == 0)

        # no negative element and stable inverse
        if is_zero == 0 and cond_val < self.th_cond:
            if self.ul_samples_per_class is not None:
                print(f"== Accepted (gap: {(est_q - self.ul_samples_per_class).abs().sum()})==")
            else:
                print("===== Accepted =====")
                print(est_q)

            self.n_infer += 1
            self.final_q += est_q

        if is_best:
            self.best_acc = max(val_top1, self.best_acc)
        self.model.train()
