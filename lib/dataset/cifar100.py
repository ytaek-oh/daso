import logging

import numpy as np
import torchvision
from yacs.config import CfgNode

from .base import BaseNumpyDataset
from .transform import build_transforms
from .utils import make_imbalance, map_dataset, split_trainval, x_u_split


def build_cifar100_dataset(cfg: CfgNode) -> tuple():
    # fmt: off
    root = cfg.DATASET.ROOT
    algorithm = cfg.ALGORITHM.NAME
    num_l_head = cfg.DATASET.CIFAR100.NUM_LABELED_HEAD
    num_ul_head = cfg.DATASET.CIFAR100.NUM_UNLABELED_HEAD
    imb_factor_l = cfg.DATASET.CIFAR100.IMB_FACTOR_L
    imb_factor_ul = cfg.DATASET.CIFAR100.IMB_FACTOR_UL
    num_valid = cfg.DATASET.NUM_VALID
    reverse_ul_dist = cfg.DATASET.REVERSE_UL_DISTRIBUTION
    seed = cfg.SEED
    # fmt: on

    logger = logging.getLogger()
    l_train = map_dataset(torchvision.datasets.CIFAR100(root, True, download=True))
    cifar100_test = map_dataset(torchvision.datasets.CIFAR100(root, False, download=True))

    # train - valid set split
    cifar100_valid = None
    if num_valid > 0:
        l_train, cifar100_valid = split_trainval(l_train, num_valid, seed=seed)

    # unlabeled sample generation unber SSL setting
    ul_train = None
    l_train, ul_train = x_u_split(l_train, num_l_head, num_ul_head, seed=seed)
    if algorithm == "Supervised":
        ul_train = None

    # whether to shuffle the class order
    num_classes = len(np.unique(l_train["labels"]))
    assert num_classes == cfg.MODEL.NUM_CLASSES
    class_inds = list(range(num_classes))

    # make synthetic imbalance for labeled set
    if imb_factor_l > 1:
        l_train, class_inds = make_imbalance(
            l_train, num_l_head, imb_factor_l, class_inds, seed=seed
        )

    # make synthetic imbalance for unlabeled set
    if ul_train is not None and imb_factor_ul > 1:
        ul_train, class_inds = make_imbalance(
            ul_train,
            num_ul_head,
            imb_factor_ul,
            class_inds,
            reverse_ul_dist=reverse_ul_dist,
            seed=seed
        )

    l_trans, ul_trans, eval_trans = build_transforms(cfg, "cifar100")

    if ul_train is not None:
        # concat purely labeled and unlabeled dataset as unlabeled dataset
        ul_train = CIFAR100Dataset(ul_train, transforms=ul_trans)

    l_train = CIFAR100Dataset(l_train, transforms=l_trans)
    if cifar100_valid is not None:
        cifar100_valid = CIFAR100Dataset(cifar100_valid, transforms=eval_trans)
    cifar100_test = CIFAR100Dataset(cifar100_test, transforms=eval_trans)

    logger.info("class distribution of labeled dataset")
    logger.info(
        ", ".join("idx{}: {}".format(item[0], item[1]) for item in l_train.num_samples_per_class)
    )
    logger.info(
        "=> number of labeled data: {}\n".format(
            sum([item[1] for item in l_train.num_samples_per_class])
        )
    )
    if ul_train is not None:
        logger.info("class distribution of unlabeled dataset")
        logger.info(
            ", ".join(
                ["idx{}: {}".format(item[0], item[1]) for item in ul_train.num_samples_per_class]
            )
        )
        logger.info(
            "=> number of unlabeled data: {}\n".format(
                sum([item[1] for item in ul_train.num_samples_per_class])
            )
        )

    return l_train, ul_train, cifar100_valid, cifar100_test


class CIFAR100Dataset(BaseNumpyDataset):

    def __init__(self, *args, **kwargs):
        super(CIFAR100Dataset, self).__init__(*args, **kwargs)
