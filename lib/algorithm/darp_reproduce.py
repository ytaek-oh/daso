import math
import os

import numpy as np
import torch
from scipy import optimize

from lib.dataset.utils import get_data_config, get_imb_num


def get_target_dist(cfg, to_prob=False, device=None):
    data_cfg = get_data_config(cfg)

    num_l_head = data_cfg.NUM_LABELED_HEAD
    num_ul_head = data_cfg.NUM_UNLABELED_HEAD
    imb_factor_l = data_cfg.IMB_FACTOR_L
    imb_factor_ul = data_cfg.IMB_FACTOR_UL
    reverse_ul = cfg.DATASET.REVERSE_UL_DISTRIBUTION

    ul_samples_per_class = get_imb_num(
        num_ul_head,
        imb_factor_ul,
        num_classes=cfg.MODEL.NUM_CLASSES,
        reverse=reverse_ul,
        normalize=False
    ) if imb_factor_ul > 0 else [100000]  # stl10

    is_dist_equal = (imb_factor_l == imb_factor_ul) and (not reverse_ul)
    if not is_dist_equal:
        # load from pre-computed (estimated) unlabeled distribution
        estim_path = cfg.ALGORITHM.DARP.EST
        est_name = f"{cfg.DATASET.NAME}_l_{num_l_head}_{imb_factor_l}_" + \
            f"ul_{num_ul_head}_{imb_factor_ul}_"
        if reverse_ul:
            est_name += "rev_"
        est_name += "estim.npyz"

        est_dist = np.load(os.path.join(estim_path, est_name))
        target_dist = sum(ul_samples_per_class) * est_dist / np.sum(est_dist)
        print("loaded estimated distribution from: {}".format(os.path.join(estim_path, est_name)))
        print("[DARP] The scaled distribution is as following:")
        print(target_dist)
    else:
        # assume labeled distribution equals unlabeled distribution
        target_dist = ul_samples_per_class
    target_dist = torch.Tensor(target_dist).float()  # cpu

    if to_prob:
        target_dist = target_dist / target_dist.sum()
    if device is not None:
        target_dist = target_dist.to(device)

    return target_dist


def f(x, a, b, c, d):
    """https://github.com/bbuing9/DARP"""
    return np.sum(a * b * np.exp(-1 * x / c)) - d


class DARP:
    """This code is constructed (mostly copied) based on pytorch implementation of DARP.
    original repository: https://github.com/bbuing9/DARP"""

    def __init__(self, cfg, ul_dataset):
        # darp configurations
        self.warmup_ratio = cfg.ALGORITHM.DARP.WARMUP_RATIO  # warm up ratio (=total_iters * ratio)
        self.per_iters = cfg.ALGORITHM.DARP.PER_ITERS  # periods for updating pseudo labels (iters)
        self.alpha = cfg.ALGORITHM.DARP.ALPHA
        self.num_darp_iters = cfg.ALGORITHM.DARP.NUM_DARP_ITERS

        # other configurations
        self.device = cfg.GPU_ID
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.max_iters = cfg.SOLVER.MAX_ITER
        self.target_dist = get_target_dist(cfg)

        # init pseudo label distributions
        self.pseudo_orig = torch.ones(len(ul_dataset), self.num_classes) / self.num_classes
        self.pseudo_refine = torch.ones(len(ul_dataset), self.num_classes) / self.num_classes

        self.total_steps = 0
        self.data_size = len(ul_dataset)

    def step(self, pl_input, data_indices):
        self.total_steps += 1

        # Update the saved predictions with current one
        self.pseudo_orig[data_indices, :] = pl_input.data.cpu()

        if self.total_steps <= int(self.max_iters * self.warmup_ratio):
            # warm-up stage; no apply darp
            return pl_input

        if self.total_steps % self.per_iters == 0:
            # update pseudo labels using DARP
            pseudo_orig_backup = self.pseudo_orig.clone()

            targets_u, weights_u = self.estimate_pseudo()
            scale_term = targets_u * weights_u.reshape(1, -1)
            scaled_orig = self.pseudo_orig * scale_term + 1e-6
            self.pseudo_orig = scaled_orig / scaled_orig.sum(dim=1, keepdim=True)
            opt_res = self.opt_solver()

            # Updated pseudo-labels are saved
            self.pseudo_refine = opt_res

            # Select
            pl_darp = opt_res[data_indices].detach().to(self.device)
            self.pseudo_orig = pseudo_orig_backup
        else:
            pl_darp = self.pseudo_refine[data_indices].detach().to(self.device)

        return pl_darp

    def estimate_pseudo(self):
        pseudo_labels = torch.zeros(len(self.pseudo_orig), self.num_classes)
        k_probs = torch.zeros(self.num_classes)

        for i in range(1, self.num_classes + 1):
            i = self.num_classes - i
            num_i = int(self.alpha * self.target_dist[i])
            sorted_probs, idx = self.pseudo_orig[:, i].sort(dim=0, descending=True)
            pseudo_labels[idx[:num_i], i] = 1
            k_probs[i] = sorted_probs[:num_i].sum()

        return pseudo_labels, (self.target_dist + 1e-6) / (k_probs + 1e-6)

    def opt_solver(self, num_newton=30):
        probs = self.pseudo_orig

        entropy = (-1 * probs * torch.log(probs + 1e-6)).sum(1)
        weights = (1 / entropy)
        N, K = probs.size(0), probs.size(1)

        A, w, lam, nu, r, c = probs.numpy(), weights.numpy(
        ), np.ones(N), np.ones(K), np.ones(N), self.target_dist.numpy()
        A_e = A / math.e
        X = np.exp(-1 * lam / w)
        Y = np.exp(-1 * nu.reshape(1, -1) / w.reshape(-1, 1))
        prev_Y = np.zeros(K)
        X_t, Y_t = X, Y

        for n in range(self.num_darp_iters):
            # Normalization
            denom = np.sum(A_e * Y_t, 1)
            X_t = r / denom

            # Newton method
            Y_t = np.zeros(K)
            for i in range(K):
                Y_t[i] = optimize.newton(
                    f, prev_Y[i], maxiter=num_newton, args=(A_e[:, i], X_t, w, c[i]), tol=1.0e-01
                )
            prev_Y = Y_t
            Y_t = np.exp(-1 * Y_t.reshape(1, -1) / w.reshape(-1, 1))

        denom = np.sum(A_e * Y_t, 1)
        X_t = r / denom
        M = torch.Tensor(A_e * X_t.reshape(-1, 1) * Y_t)

        return M
