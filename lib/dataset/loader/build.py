import logging

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from yacs.config import CfgNode

from typing import Tuple

from ..cifar10 import build_cifar10_dataset  # noqa
from ..cifar100 import build_cifar100_dataset  # noqa
from ..stl10 import build_stl10_dataset  # noqa
from .class_aware_sampler import ClassAwareSampler


class RandomSampler(Sampler):
    """ sampling without replacement """

    def __init__(self, data_size: int, total_samples: int) -> None:
        num_epochs = total_samples // data_size + 1
        _indices = torch.cat([torch.randperm(data_size) for _ in range(num_epochs)])
        self._indices = _indices.tolist()[:total_samples]

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def _build_loader(
    cfg: CfgNode, dataset: Dataset, *, is_train: bool = True, has_label: bool = True
) -> DataLoader:
    logger = logging.getLogger()

    sampler = None
    drop_last = is_train
    batch_size = cfg.SOLVER.IMS_PER_BATCH

    if not has_label:
        batch_size = int(batch_size * cfg.SOLVER.UNLABELED_BATCH_RATIO)

    if is_train:
        sampler_name = cfg.DATASET.SAMPLER_NAME
        if not has_label:
            sampler_name = "RandomSampler"

        max_iter = cfg.SOLVER.MAX_ITER
        total_samples = max_iter * batch_size
        if sampler_name == "RandomSampler":
            sampler = RandomSampler(len(dataset), total_samples)
        elif sampler_name == "ClassAwareSampler":
            beta = cfg.DATASET.SAMPLER_BETA
            sampler = ClassAwareSampler(dataset, total_samples, beta=beta, shuffle=True)

            logger.info(
                "ClassAwareSampler is enabled.  "
                "per_class probabilities: {}".format(
                    ", ".join(["{:.4f}".format(v) for v in sampler.per_cls_prob])
                )
            )
        else:
            raise ValueError

    # train: drop last true
    # test:  drop last false
    if (not has_label) and is_train and (cfg.ALGORITHM.NAME == "DARP_ESTIM"):
        sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.DATASET.NUM_WORKERS,
        drop_last=drop_last,
        sampler=sampler,
        shuffle=False
    )
    return data_loader


def build_data_loaders(cfg: CfgNode) -> Tuple[DataLoader]:
    builder = cfg.DATASET.BUILDER
    l_train, ul_train, val_dataset, test_dataset = eval(builder)(cfg)
    l_loader = _build_loader(cfg, l_train)

    ul_loader = None
    if ul_train is not None:
        ul_loader = _build_loader(cfg, ul_train, has_label=False)

    val_loader = None
    if val_dataset is not None:
        val_loader = _build_loader(cfg, val_dataset, is_train=False)
    test_loader = _build_loader(cfg, test_dataset, is_train=False)

    return l_loader, ul_loader, val_loader, test_loader
