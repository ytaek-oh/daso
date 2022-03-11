import numpy as np
import torchvision

from typing import Tuple, List


def map_dataset(dataset: torchvision.datasets, dtype: str = "float") -> dict:
    """CIFAR-10/100, STL dataset mapper"""
    if dtype == "float":
        dtype = np.float32
    elif dtype == "uint8":
        dtype = np.uint8
    else:
        raise ValueError("dtype {} is invalid.".format(dtype))

    mapped_data = dict()
    mapped_data["images"] = dataset.data.astype(dtype)
    if hasattr(dataset, "targets"):
        mapped_data["labels"] = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        mapped_data["labels"] = np.array(dataset.labels)
    return mapped_data


def split_trainval(trainval: np.ndarray, num_valid: int, seed: int = 0) -> Tuple[dict]:
    """
        split train and validation datasets for cifar datasets.
        randomly select class-balanced validation samples.
    """
    # set random state
    rng = np.random.RandomState(seed)

    trainval_images = trainval["images"]
    trainval_labels = trainval["labels"]

    num_classes = len(np.unique(trainval_labels))
    num_valid_cls = num_valid // num_classes

    train_inds = []
    val_inds = []
    for i in range(num_classes):
        cls_inds = np.where(trainval_labels == i)[0]
        rng.shuffle(cls_inds)
        train_inds.extend(cls_inds[num_valid_cls:])
        val_inds.extend(cls_inds[:num_valid_cls])

    train_dataset = dict(images=trainval_images[train_inds], labels=trainval_labels[train_inds])
    val_dataset = dict(images=trainval_images[val_inds], labels=trainval_labels[val_inds])
    return train_dataset, val_dataset


def split_val_from_train(trainval, num_valid_cls):
    trainval_images = trainval["images"]
    trainval_labels = trainval["labels"]

    num_classes = len(np.unique(trainval_labels))
    # num_valid_cls = num_valid // num_classes

    train_inds = []
    val_inds = []
    for i in range(num_classes):
        cls_inds = np.where(trainval_labels == i)[0]

        # disjoint
        train_inds.extend(cls_inds[num_valid_cls:])
        val_inds.extend(cls_inds[:num_valid_cls])

    train_dataset = dict(images=trainval_images[train_inds], labels=trainval_labels[train_inds])
    val_dataset = dict(images=trainval_images[val_inds], labels=trainval_labels[val_inds])
    return train_dataset, val_dataset


def x_u_split(
    train_dataset: np.ndarray,
    num_l_head: int,
    num_ul_head: int,
    seed: int = 0,
) -> Tuple[dict]:
    rng = np.random.RandomState(seed)

    images = train_dataset["images"]
    labels = train_dataset["labels"]
    num_classes = len(np.unique(labels))

    labeled_inds = []
    unlabeled_inds = []
    for label in range(num_classes):
        inds = np.where(labels == label)[0]
        rng.shuffle(inds)
        labeled_inds.extend(inds[:num_l_head])
        unlabeled_inds.extend(inds[num_l_head:num_l_head + num_ul_head])

    train_labeled = dict(images=images[labeled_inds], labels=labels[labeled_inds])
    train_unlabeled = dict(images=images[unlabeled_inds], labels=labels[unlabeled_inds])
    return train_labeled, train_unlabeled


def make_imbalance(
    dataset: np.ndarray,
    num_head: int,
    imb_factor: int,
    class_inds: List[int],
    *,
    reverse_ul_dist: bool = False,
    seed: int = 0
) -> Tuple[dict, List[int]]:
    rng = np.random.RandomState(seed)

    images = dataset["images"]
    labels = dataset["labels"]
    num_classes = len(np.unique(labels))
    inds = []

    if reverse_ul_dist:
        class_inds.reverse()

    for rank, label in enumerate(class_inds):
        cls_inds = np.where(labels == label)[0]
        rng.shuffle(cls_inds)

        num = int(num_head * ((1. / imb_factor)**(rank / (num_classes - 1.0))))
        inds.extend(cls_inds[:num])

    imb_train = dict(images=images[inds], labels=labels[inds])
    return imb_train, class_inds


def get_data_config(cfg):
    return {
        "cifar10": cfg.DATASET.CIFAR10,
        "cifar100": cfg.DATASET.CIFAR100,
        "stl10": cfg.DATASET.STL10
    }[cfg.DATASET.NAME]


def get_imb_num(num_head, imb_factor, num_classes=10, reverse=False, normalize=False):
    nums = []
    classes = list(range(num_classes))  # [0, 1, ..., 9]
    if reverse:
        classes.reverse()
    for rank in classes:
        num = int(num_head * ((1. / imb_factor)**(rank / (num_classes - 1.0))))
        nums.append(num)
    if normalize:
        nums = [np.round(num / min(nums), 1) for num in nums]
    return nums


def get_class_counts(dataset):
    """
        Sort the class counts by class index in an increasing order
        i.e., List[(2, 60), (0, 30), (1, 10)] -> np.array([30, 10, 60])
    """
    class_count = dataset.num_samples_per_class

    # sort with class indices in increasing order
    class_count.sort(key=lambda x: x[0])
    per_class_samples = np.asarray([float(v[1]) for v in class_count])
    return per_class_samples
