import argparse
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from yacs.config import CfgNode

from typing import Optional


def default_argument_parser():
    """
    Create a parser with some common arguments

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def default_setup(cfg: CfgNode, args: argparse.Namespace) -> CfgNode:
    """
    Perform some basic common setups at the beginning of a job

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.Namespace): the command line arguments to be logged

    Returns:
        cfg (CfgNode)
    """
    if cfg.ALGORITHM.NAME == "cRT":
        target_dir = cfg.ALGORITHM.CRT.TARGET_DIR
        if "_seed_" in target_dir:
            target_seed = target_dir.split("_seed_")[1]
            if "_" in target_seed:
                target_seed = target_seed.split("_")[0]
        else:
            target_seed = cfg.SEED
        cfg.SEED = int(target_seed)
    seed = _set_seed(None if cfg.SEED < 0 else cfg.SEED)
    experiment_name = _get_experiment_name(cfg, seed)

    output_dir = os.path.join(cfg.OUTPUT_DIR, experiment_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger = _setup_logger()

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                Path(args.config_file).open().read()
            )
        )

    # reproducability
    torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN_DETERMINISTIC

    cfg.OUTPUT_DIR = output_dir
    cfg.SEED = seed

    # logger.info("Running with full config:\n{}".format(cfg))
    if output_dir:
        path = os.path.join(output_dir, "config.yaml")
        with Path(path).open("w") as f:
            f.write(cfg.dump())
    logger.info("Full config saved to {}".format(cfg.OUTPUT_DIR))

    return cfg


def _get_experiment_name(cfg: CfgNode, seed: int) -> str:
    """
    Set experiment name based on config

    Args:
        cfg (CfgNode): the full config to be used
        seed (int): random seed to be used
    Returns:
        exp_name (str)
    """
    algorithm = cfg.ALGORITHM.NAME
    with_darp = cfg.ALGORITHM.DARP.APPLY
    dataset_name = cfg.DATASET.NAME
    if dataset_name == "cifar10":
        data_cfg = cfg.DATASET.CIFAR10
    elif dataset_name == "cifar100":
        data_cfg = cfg.DATASET.CIFAR100
    elif dataset_name == "stl10":
        data_cfg = cfg.DATASET.STL10
    else:
        raise ValueError

    num_l_head = data_cfg.NUM_LABELED_HEAD
    num_ul_head = data_cfg.NUM_UNLABELED_HEAD
    imb_l = data_cfg.IMB_FACTOR_L
    imb_ul = data_cfg.IMB_FACTOR_UL
    reverse_ul = cfg.DATASET.REVERSE_UL_DISTRIBUTION

    l_loss = cfg.MODEL.LOSS.LABELED_LOSS
    memo = cfg.MEMO

    exp_names = []
    alg_name = algorithm
    if with_darp:
        alg_name = algorithm + "_" + "DARP"

    if "CReST" in algorithm:
        if cfg.MODEL.DIST_ALIGN.APPLY:
            alg_name = algorithm + "_Plus"  # CReST+

    # specify algorithm name with DARP, CReST+
    exp_names.append(f"{alg_name}_{dataset_name}")

    # labeled info
    exp_names.append(f"l_{num_l_head}_{imb_l}")

    # unlabeled_info
    if (algorithm != "Supervised") and (dataset_name != "stl10"):
        exp_names.append(f"ul_{num_ul_head}_{imb_ul}")
        if reverse_ul:
            exp_names.append("ul_rev")
    if dataset_name == "stl10":
        exp_names.append("ul_unknown")

    if l_loss != "CrossEntropyLoss":
        exp_names.append(f"{l_loss}")

    exp_names.append(f"seed_{seed}")

    if memo:
        exp_names.append(memo)
    return "_".join(exp_names)


def _setup_logger(name: str = "semi"):
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    return logger


def _set_seed(seed: Optional[int] = None) -> int:
    """
    Set the random seed for the RNG in torch, numpy and python.
    Args:
        seed (int): if None, will use a strong random seed.

    Returns:
        seed (int): proper random seed
    """
    if seed is None:
        # random seed generation
        seed = (
            os.getpid() + int(datetime.now().strftime("%S%f")) +  # noqa
            int.from_bytes(os.urandom(2), "big")
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger = logging.getLogger(__name__)
    logger.info("Using a random seed {}".format(seed))

    return seed
