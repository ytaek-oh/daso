import time

import numpy as np
import torch
from yacs.config import CfgNode

from lib.dataset.base import BaseNumpyDataset
from lib.dataset.loader.build import _build_loader
from lib.models import EMAModel

from .fixmatch import FixMatch


class FixMatchCReST(FixMatch):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        # CReST options
        self.gen_period_steps = cfg.ALGORITHM.CREST.GEN_PERIOD_STEPS
        self.t_min = cfg.ALGORITHM.CREST.TMIN
        self.with_progressive = cfg.ALGORITHM.CREST.PROGRESSIVE_ALIGN

        # unlabeled dataset configuration
        ul_dataset = self.ul_loader.dataset
        ul_test_dataset = BaseNumpyDataset(
            ul_dataset.select_dataset(),
            transforms=self.test_loader.dataset.transforms,
            is_ul_unknown=ul_dataset.is_ul_unknown
        )
        self.ul_test_loader = _build_loader(
            self.cfg, ul_test_dataset, is_train=False, has_label=False
        )

        # save init stats
        l_dataset = self.l_loader.dataset
        self.init_l_data, self.l_transforms = l_dataset.select_dataset(return_transforms=True)
        self.current_l_dataset = l_dataset

        crest_alpha = cfg.ALGORITHM.CREST.ALPHA
        self.mu_per_cls = torch.pow(
            self.current_label_dist(device="cpu", normalize="max").clone(), (1 / crest_alpha)
        )

        # rebuild scheduler; fixing max steps to GEN_PERIOD_STEPS
        self._rebuild_lr_scheduler(self.optimizer)

    def eval_ul_dataset(self):
        self.logger.info("evaluating ul data as test set...")
        ul_dataset = self.ul_loader.dataset
        ul_preds = torch.zeros(len(ul_dataset), self.num_classes)

        model = self.ema_model
        model.eval()
        with torch.no_grad():
            for i, (images, _, inds) in enumerate(self.ul_test_loader):
                if torch.cuda.is_available():
                    images = images.to(self.device)
                outputs = model(images, is_train=False)
                ul_preds[inds, :] = outputs.softmax(dim=1).detach().data.cpu()
        model.train()

        return ul_preds

    def _rebuild_models(self):
        model = self.build_model(self.cfg)
        if torch.cuda.is_available():
            model = model.to(self.device)
        self.model = model
        self.ema_model = EMAModel(
            self.model,
            self.cfg.MODEL.EMA_DECAY,
            self.cfg.MODEL.EMA_WEIGHT_DECAY,
            device=self.device,
            resume=self.resume
        )

    def _rebuild_optimizer(self, model):
        self.optimizer = self.build_optimizer(self.cfg, model)

    def _rebuild_lr_scheduler(self, optimizer):
        self.scheduler = self.build_lr_scheduler(
            self.cfg, optimizer, override_max_iter=self.gen_period_steps
        )

    def _rebuild_labeled_dataset(self):
        ul_preds = self.eval_ul_dataset()
        conf, pred_class = torch.max(ul_preds, dim=1)

        selected_inds = []
        selected_labels = []
        for i in range(self.num_classes):
            inds = torch.where(pred_class == i)[0]
            if len(inds) == 0:
                continue
            num_selected = int(self.mu_per_cls[self.num_classes - (i + 1)] * len(inds))
            if num_selected < 1:
                continue

            sorted_inds = torch.argsort(conf[inds], descending=True)
            selected = inds[sorted_inds[:num_selected]]

            selected_inds.extend(selected.tolist())
            selected_labels.extend([i] * num_selected)

        ul_dataset = self.ul_loader.dataset
        ul_data_np = ul_dataset.select_dataset(indices=selected_inds, labels=selected_labels)

        new_data_dict = {
            k: np.concatenate([self.init_l_data[k], ul_data_np[k]], axis=0)
            for k in self.init_l_data.keys()
        }
        new_l_dataset = BaseNumpyDataset(new_data_dict, transforms=self.l_transforms)
        new_loader = _build_loader(self.cfg, new_l_dataset)

        self.current_l_dataset = new_l_dataset
        self._l_iter = iter(new_loader)

        # for logging
        per_class_sample = self.current_label_dist(device="cpu").tolist()
        self.logger.info("Categorical distributions of labeled dataset:")
        self.logger.info(per_class_sample)
        self.logger.info(
            "imb ratio: {:.2f}".format(
                per_class_sample[0] / per_class_sample[self.num_classes - 1]
            )
        )
        print()

    # starting new generation -> see base_algoritmh.py
    def new_generation(self):
        print()
        self.logger.info(
            "{} iters -> {}-th generation".format(
                self.iter + 1, (self.iter + 1) // self.gen_period_steps + 1
            )
        )
        self._rebuild_labeled_dataset()

        self._rebuild_models()
        self._rebuild_optimizer(self.model)
        self._rebuild_lr_scheduler(self.optimizer)

    @property
    def max_gen(self):
        max_iter = self.cfg.SOLVER.MAX_ITER
        assert max_iter % self.gen_period_steps == 0
        return max_iter // self.gen_period_steps

    @property
    def current_gen(self):
        return (self.iter) // self.gen_period_steps + 1

    @property
    def gradual_temp(self):
        factor = (self.current_gen - 1) / (self.max_gen - 1)
        return 1.0 - factor + factor * self.t_min

    def current_label_dist(self, **kwargs):
        return self.get_label_dist(dataset=self.current_l_dataset, **kwargs)

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

        input_concat = torch.cat([l_images, ul_weak, ul_strong], 0)
        logits_concat = self.model(input_concat)

        l_logits = logits_concat[:num_labels]

        # logit adjustment
        if self.with_la:
            per_class_samples = self.current_label_dist(device=self.device, normalize="sum")
            l_logits += (self.tau * per_class_samples.view(1, -1).log())

        cls_loss = self.l_loss(l_logits, labels)
        loss_dict.update({"loss_cls": cls_loss})

        # unlabeled loss
        logits_weak, logits_strong = logits_concat[num_labels:].chunk(2)
        p = logits_weak.detach_().softmax(dim=1)  # soft pseudo labels

        # DA
        if self.with_align:
            da_t = self.gradual_temp if self.with_progressive else None
            with torch.no_grad():
                p = self.dist_align(p, temperature=da_t)

        confidence, pred_class = torch.max(p, dim=1)
        loss_weight = confidence.ge(self.conf_thres)

        cons_loss = self.ul_loss(
            logits_strong,
            pred_class,
            pl_count=None,
            weight=loss_weight,
            avg_factor=ul_weak.size(0)
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
        ema_decay = self.ema_model.update(
            self.model, step=self.iter % self.gen_period_steps, current_lr=current_lr
        )

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

        self.meters.put_scalar("crest/DA_temperature", self.gradual_temp, show_avg=False)
        self.meters.put_scalar("crest/current_gen", self.current_gen, show_avg=False)

        # make a log for accuracy and losses
        self._write_metrics(metrics_dict, n=l_images.size(0), prefix="train")
