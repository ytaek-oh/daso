import logging

import torchvision
from yacs.config import CfgNode

from .base import BaseNumpyDataset
from .transform import build_transforms
from .utils import make_imbalance, map_dataset, split_trainval, split_val_from_train, x_u_split


def build_cifar10_dataset(cfg: CfgNode) -> tuple():
    # fmt: off
    root = cfg.DATASET.ROOT
    algorithm = cfg.ALGORITHM.NAME
    num_l_head = cfg.DATASET.CIFAR10.NUM_LABELED_HEAD
    num_ul_head = cfg.DATASET.CIFAR10.NUM_UNLABELED_HEAD
    imb_factor_l = cfg.DATASET.CIFAR10.IMB_FACTOR_L
    imb_factor_ul = cfg.DATASET.CIFAR10.IMB_FACTOR_UL
    num_valid = cfg.DATASET.NUM_VALID
    reverse_ul_dist = cfg.DATASET.REVERSE_UL_DISTRIBUTION

    num_classes = cfg.MODEL.NUM_CLASSES
    seed = cfg.SEED
    # fmt: on

    logger = logging.getLogger()
    l_train = map_dataset(torchvision.datasets.CIFAR10(root, True, download=True))
    cifar10_test = map_dataset(torchvision.datasets.CIFAR10(root, False, download=True))

    # train - valid set split
    cifar10_valid = None
    if num_valid > 0:
        l_train, cifar10_valid = split_trainval(l_train, num_valid, seed=seed)

    # unlabeled sample generation unber SSL setting
    ul_train = None
    l_train, ul_train = x_u_split(l_train, num_l_head, num_ul_head, seed=seed)
    if algorithm == "Supervised":
        ul_train = None

    # whether to shuffle the class order
    class_inds = list(range(num_classes))

    # make synthetic imbalance for labeled set
    if imb_factor_l > 1:
        l_train, class_inds = make_imbalance(
            l_train, num_l_head, imb_factor_l, class_inds, seed=seed
        )

    if cfg.ALGORITHM.NAME == "DARP_ESTIM":
        # held-out validation images subtracting from train images (DARP estimation stage)
        num_l_tail = int(num_l_head * 1. / imb_factor_l)
        num_holdout = cfg.ALGORITHM.DARP_ESTIM.PER_CLASS_VALID_SAMPLES
        if num_l_tail > 10:
            l_train, cifar10_valid = split_val_from_train(l_train, num_holdout)
        else:
            logger.info(
                f"Tail class training examples ({num_l_tail}) are not sufficient. "
                f"for constructing hold-out validation images ({num_holdout}). "
                "Extracting from original validation set."
            )
            _, cifar10_valid = split_val_from_train(cifar10_valid, num_holdout)

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

    l_trans, ul_trans, eval_trans = build_transforms(cfg, "cifar10")

    if ul_train is not None:
        ul_train = CIFAR10Dataset(ul_train, transforms=ul_trans)

    l_train = CIFAR10Dataset(l_train, transforms=l_trans)
    if cifar10_valid is not None:
        cifar10_valid = CIFAR10Dataset(cifar10_valid, transforms=eval_trans)
    cifar10_test = CIFAR10Dataset(cifar10_test, transforms=eval_trans)

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

    return l_train, ul_train, cifar10_valid, cifar10_test


class CIFAR10Dataset(BaseNumpyDataset):

    def __init__(self, *args, **kwargs):
        super(CIFAR10Dataset, self).__init__(*args, **kwargs)
