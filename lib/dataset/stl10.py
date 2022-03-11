import logging

from torchvision.datasets import STL10
from yacs.config import CfgNode

from .base import BaseNumpyDataset
from .transform import build_transforms
from .utils import make_imbalance, map_dataset, split_trainval, split_val_from_train


def build_stl10_dataset(cfg: CfgNode) -> tuple():
    # fmt: off
    root = cfg.DATASET.ROOT
    algorithm = cfg.ALGORITHM.NAME
    num_l_head = cfg.DATASET.STL10.NUM_LABELED_HEAD
    imb_factor_l = cfg.DATASET.STL10.IMB_FACTOR_L
    num_valid = cfg.DATASET.NUM_VALID

    num_classes = cfg.MODEL.NUM_CLASSES
    seed = cfg.SEED
    # fmt: on

    logger = logging.getLogger()
    l_train = map_dataset(STL10(root, split="train", download=True))
    ul_train = map_dataset(STL10(root, split="unlabeled", download=True))
    stl10_test = map_dataset(STL10(root, split="test"))

    # train - valid set split
    stl10_valid = None
    if num_valid > 0:
        l_train, stl10_valid = split_trainval(l_train, num_valid, seed=seed)

    # unlabeled sample generation unber SSL setting
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
        # construct valid images subtracting from train images (DARP estimation stage)
        num_l_tail = int(num_l_head * 1. / imb_factor_l)
        num_holdout = cfg.ALGORITHM.DARP_ESTIM.PER_CLASS_VALID_SAMPLES
        if num_l_tail > 10:
            l_train, stl10_valid = split_val_from_train(l_train, num_holdout)
        else:
            logger.info(
                "Tail class training examples ({}) are not sufficient. "
                "for constructing hold-out validation images ({}). "
                "Extracting from original validation set.".format(num_l_tail, num_holdout)
            )
            _, stl10_valid = split_val_from_train(stl10_valid, num_holdout)

    l_trans, ul_trans, eval_trans = build_transforms(cfg, "stl10")

    l_train = STL10Dataset(l_train, transforms=l_trans)
    if ul_train is not None:
        ul_train = STL10Dataset(ul_train, transforms=ul_trans, is_ul_unknown=True)

    if stl10_valid is not None:
        stl10_valid = STL10Dataset(stl10_valid, transforms=eval_trans)
    stl10_test = STL10Dataset(stl10_test, transforms=eval_trans)

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
        logger.info("Number of unlabeled data: {}\n".format(len(ul_train)))

    return l_train, ul_train, stl10_valid, stl10_test


class STL10Dataset(BaseNumpyDataset):

    def __init__(self, *args, **kwargs):
        super(STL10Dataset, self).__init__(*args, **kwargs)
