import json
import os

import torch
import torch.nn.functional as F
from yacs.config import CfgNode

from lib.dataset.utils import get_class_counts
from lib.engine import BaseTrainer
from lib.models.losses import Accuracy
from lib.utils import Meters, get_last_n_median
from lib.utils.dist_logger import DistributionLogger


class BaseAlgorithm(BaseTrainer):
    """class for BaseAlgorithm"""

    def __init__(self, cfg: CfgNode):
        model = self.build_model(cfg)
        if torch.cuda.is_available():
            model = model.to(cfg.GPU_ID)
        optimizer = self.build_optimizer(cfg, model)
        l_loader, ul_loader, valid_loader, test_loader = self.build_data_loaders(cfg)
        super().__init__(
            cfg,
            model,
            optimizer,
            l_loader,
            ul_loader=ul_loader,
            valid_loader=valid_loader,
            test_loader=test_loader
        )
        self.num_classes = cfg.MODEL.NUM_CLASSES

        is_ul_unknown = cfg.DATASET.NAME == "stl10"
        self.dist_logger = DistributionLogger(
            self.meters, num_classes=self.num_classes, is_ul_unknown=is_ul_unknown
        )

        self.gen_period_steps = cfg.ALGORITHM.CREST.GEN_PERIOD_STEPS  # for CReST
        self.is_warmed = False  # for build LDAM loss

        # label distribution
        self.p_data = self.get_label_dist(normalize="sum", device=self.device)

        # logit adjustment
        self.with_la = cfg.ALGORITHM.LOGIT_ADJUST.APPLY
        self.tau = cfg.ALGORITHM.LOGIT_ADJUST.TAU

    def get_label_dist(self, dataset=None, normalize=None, device=None):
        """
            normalize: ["sum", "max"]
        """
        if dataset is None:
            dataset = self.l_loader.dataset

        class_counts = torch.from_numpy(get_class_counts(dataset)).float()
        if device is not None:
            class_counts = class_counts.to(device)

        if normalize:
            assert normalize in ["sum", "max"]
            if normalize == "sum":
                return class_counts / class_counts.sum()
            if normalize == "max":
                return class_counts / class_counts.max()
        return class_counts

    def train(self):
        self.logger.info(f"Starting training from iteration {self.start_iter}")
        self.model.train()

        for self.iter in range(self.start_iter, self.max_iter):
            if (
                self.cfg.MODEL.LOSS.WITH_LABELED_COST_SENSITIVE
                and (self.iter + 1) >= self.cfg.MODEL.LOSS.WARMUP_ITERS and not self.is_warmed
            ):
                # warmup, LDAM-DRW (deferred reweight)
                self.is_warmed = True
                self.l_loss = self.build_labeled_loss(self.cfg, warmed_up=True)

            # one step of forward path and backprop
            self.run_step()

            # increase the meter's iteration
            self.meters.step()

            # eval period
            if ((self.iter + 1) % self.cfg.PERIODS.EVAL == 0):
                self.evaluate(self.model)
                self.dist_logger.write()

            # periodically save checkpoints
            if (
                self.cfg.PERIODS.CHECKPOINT > 0
                and (self.iter + 1) % self.cfg.PERIODS.CHECKPOINT == 0
            ):
                save_ema_model = self.with_ul
                if self.cfg.ALGORITHM.NAME == "DARP_ESTIM":
                    save_ema_model = False
                self.save_checkpoint(save_ema_model=save_ema_model)

            # print logs
            if (((self.iter + 1) % self.cfg.PERIODS.LOG == 0 or (self.iter + 1) == self.max_iter)):
                assert self.cfg.PERIODS.EVAL == self.cfg.PERIODS.LOG
                for writer in self.writers:
                    writer.write(self.meters)
                self.meters.reset()

            # start new generation after evaluation!
            if (self.iter + 1) % self.gen_period_steps == 0:
                crest_names = ["ReMixMatchCReST", "FixMatchCReST"]
                with_crest = self.cfg.ALGORITHM.NAME in crest_names
                # new generation except for the last iteration
                if with_crest and (self.iter + 1) < self.max_iter:
                    self.new_generation()
        print()
        print()
        print()

        prefixes = ["valid/top1", "test/top1"]
        self.logger.info("Median 20 Results:")
        self.logger.info(
            ", ".join(
                f"{k}_median (20): {get_last_n_median(v, n=20):.2f}"
                for k, v in self.eval_history.items() if k in prefixes
            )
        )
        print()
        prefixes = ["valid/top1_la", "test/top1_la"]
        self.logger.info("Median 20 Results:")
        self.logger.info(
            ", ".join(
                f"Logit adjusted {k}_median (20): {get_last_n_median(v, n=20):.2f}"
                for k, v in self.eval_history.items() if k in prefixes
            )
        )
        print()

        # final checkpoint
        self.save_checkpoint(save_ema_model=self.with_ul)

        # test top1 and median print
        print()
        save_path = self.cfg.OUTPUT_DIR
        with open(os.path.join(save_path, "results.json"), "w") as f:
            eval_history = {k: v for k, v in self.eval_history.items()}
            f.write(json.dumps(eval_history, indent=4, sort_keys=True))
        self.logger.info(f"final results (results.json) saved on: {save_path}.")

        for writer in self.writers:
            writer.close()

    def run_step(self):
        raise NotImplementedError

    def evaluate(self, model):
        valid_results = None
        if self.valid_loader:
            valid_results = self.eval_loop(model, self.valid_loader, prefix="valid")
            self.meters.put_scalars(valid_results, show_avg=False, prefix="valid")

        test_results = None
        if self.test_loader and self.cfg.EVAL_ON_TEST_SET:
            test_results = self.eval_loop(model, self.test_loader, prefix="test")
            self.meters.put_scalars(test_results, show_avg=False, prefix="test")

        # calculate last 20 median
        metrics = {}
        last_n_evals = [20]
        prefixes = ["valid/top1", "valid/top1_la", "test/top1", "test/top1_la"]
        for _prefix in prefixes:
            if _prefix in self.eval_history.keys():
                for last_n in last_n_evals:
                    metric_key = f"{_prefix}_median{last_n}"
                    median_acc = get_last_n_median(self.eval_history[_prefix], n=last_n)
                    metrics[metric_key] = median_acc
                    self.eval_history[metric_key].append(median_acc)

        if len(metrics.keys()) > 0:
            self.meters.put_scalars(metrics, show_avg=False)  # log median accuracies

        return test_results

    def eval_loop(self, model, data_loader, *, prefix: str = "valid") -> float:
        # local metric and meters
        accuracy = Accuracy(self.num_classes)
        meters = Meters()

        model.eval()
        log_classwise = self.cfg.MISC.LOG_CLASSWISE
        with torch.no_grad():
            for i, (images, target, _) in enumerate(data_loader):
                metrics = {}
                if torch.cuda.is_available():
                    images = images.to(self.device)
                    target = target.to(self.device).long()

                outputs = model(images, is_train=False)  # logits
                batch_size = images.size(0)

                # compute metrics using original logits
                loss = F.cross_entropy(outputs, target, reduction="none").mean()
                top1, top5 = accuracy(outputs, target, log_classwise=log_classwise)
                metrics.update({"cost": loss.item(), "top1": top1, "top5": top5})

                # adjust logits in test-time and compute metrics again
                outputs_la = outputs - self.p_data.view(1, -1).log()
                loss_la = F.cross_entropy(outputs_la, target, reduction="none").mean()
                top1_la, top5_la = accuracy(
                    outputs_la, target, log_classwise=log_classwise, prefix="logit_adjusted"
                )
                metrics.update({"cost_la": loss_la.item(), "top1_la": top1_la, "top5_la": top5_la})
                meters.put_scalars(metrics, n=batch_size)

        # log classwise accuracy
        if log_classwise:
            self.meters.put_scalars(
                accuracy.classwise, show_avg=False, prefix=prefix + "_classwise"
            )

        # aggregate the metrics and log
        results = meters.get_latest_scalars_with_avg()
        self.eval_history[prefix + "/top1"].append(results["top1"])
        self.eval_history[prefix + "/top1_la"].append(results["top1_la"])

        model.train()

        return results

    def _write_metrics(self, metrics_dict: dict, n: int = 1, prefix: str = ""):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics. i.e., losses and accuracy
        """
        # from tensor to float
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # reduce loss
        loss_dict = {k: v for k, v in metrics_dict.items() if "loss" in k}
        total_losses_reduced = sum(loss for loss in loss_dict.values())
        if len(loss_dict) > 1:
            self.meters.put_scalar(name=f"{prefix}/total_loss", val=total_losses_reduced, n=n)
        self.meters.put_scalars(metrics_dict, n=n, prefix=prefix)
