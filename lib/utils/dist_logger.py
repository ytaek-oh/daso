from collections import defaultdict

import numpy as np
import torch


class DistributionLogger:

    def __init__(self, meters, num_classes=10, is_ul_unknown=False):
        self.meters = meters
        self.num_classes = num_classes
        self.is_ul_unknown = is_ul_unknown
        self.reset()

        self._pl_total_list = []
        self._pl_dist = [0 for i in range(self.num_classes)]

    def reset(self):
        self._bank = defaultdict(list)

    def accumulate(self, dist_dict):
        for key, val in dist_dict.items():
            self._bank[key].append(val)

    def push_pl_list(self, pl_list):
        # for DASO
        self._pl_total_list.append(pl_list)

    def update_pl_dist(self):
        # for DASO
        pl_total_list = torch.cat(self._pl_total_list, 0)
        for class_ind in range(self.num_classes):
            pl_row_inds = torch.where(pl_total_list == class_ind)[0]
            self._pl_dist[class_ind] = len(pl_row_inds)
        self._pl_total_list = []

    def get_pl_dist(self, normalize=True):
        # for DASO
        if isinstance(self._pl_dist, list):
            pl_dist = torch.Tensor(self._pl_dist).float()
        else:
            pl_dist = self._pl_dist.float()
        if normalize:
            pl_dist = pl_dist / pl_dist.sum()
        return pl_dist

    def write(self, save_confusion_matrix=False):
        # per-evaluation period
        valid_keys = []
        for k, v in self._bank.items():
            if v is None or v[0] is None:
                continue
            self._bank[k] = torch.cat(v, 0)
            valid_keys.append(k)
        if len(valid_keys) < 1:
            return

        with_sem_pl = "sem_pl" in valid_keys
        with_lin_pl = "linear_pl" in valid_keys

        # to be logged
        gt_count_dict = {}

        # pseudo-label (final)
        pl_count_dict = {}
        pl_recall_dict = {}
        pl_prec_dict = {}
        pl_gain_dict = {}

        # linear pseudo-label (optional)
        lin_pl_count_dict = {}
        lin_recall_dict = {}
        lin_prec_dict = {}
        lin_pl_gain_dict = {}

        # semantic pseudo-label (optional)
        sem_pl_count_dict = {}
        sem_recall_dict = {}
        sem_prec_dict = {}
        sem_pl_gain_dict = {}

        for class_ind in range(self.num_classes):
            # log gt label counts
            gt_row_inds = torch.where(self._bank["gt_labels"] == class_ind)[0]
            gt_count_dict[str(class_ind)] = len(gt_row_inds)

            if not self.is_ul_unknown:
                ul_gt_list = self._bank["ul_labels"]
                ul_gt_row_inds = torch.where(ul_gt_list == class_ind)[0]

            if "pseudo_labels" in valid_keys:
                # log pl distribution
                pl_row_inds = torch.where(self._bank["pseudo_labels"] == class_ind)[0]
                pl_count_dict[str(class_ind)] = len(pl_row_inds)

                if (not self.is_ul_unknown):
                    pl_gt = self._bank["pseudo_labels"][ul_gt_row_inds]
                    pl_acc = (pl_gt == class_ind).float().mean().item()
                    pl_recall_dict[str(class_ind)] = pl_acc
                    pl_gain_dict[str(class_ind)] = float(len(pl_row_inds)) / len(ul_gt_row_inds)

                    if len(pl_row_inds) > 0:
                        prec_gt = ul_gt_list[pl_row_inds]
                        _prec = (prec_gt == class_ind).float().mean().item()
                    else:
                        _prec = -1
                    pl_prec_dict[str(class_ind)] = _prec
                # end if
            # end if

            if with_lin_pl:
                lin_pl_list = self._bank["linear_pl"]
                lin_pl_row_inds = torch.where(lin_pl_list == class_ind)[0]
                lin_pl_count_dict[str(class_ind)] = len(lin_pl_row_inds)

            if with_sem_pl:
                sem_pl_list = self._bank["sem_pl"]
                sem_pl_row_inds = torch.where(sem_pl_list == class_ind)[0]
                sem_pl_count_dict[str(class_ind)] = len(sem_pl_row_inds)

            if not self.is_ul_unknown:
                if with_lin_pl:
                    _rec = (lin_pl_list[ul_gt_row_inds] == class_ind).float().mean().item()
                    lin_recall_dict[str(class_ind)] = _rec
                    lin_pl_gain_dict[str(class_ind)
                                     ] = float(len(lin_pl_row_inds)) / len(ul_gt_row_inds)

                    # prec
                    if len(lin_pl_row_inds) > 0:
                        _prec = (ul_gt_list[lin_pl_row_inds] == class_ind).float().mean().item()
                    else:
                        _prec = -1
                    lin_prec_dict[str(class_ind)] = _prec

                if with_sem_pl:
                    _rec = (sem_pl_list[ul_gt_row_inds] == class_ind).float().mean().item()
                    sem_recall_dict[str(class_ind)] = _rec
                    sem_pl_gain_dict[str(class_ind)
                                     ] = float(len(sem_pl_row_inds)) / len(ul_gt_row_inds)

                    if len(sem_pl_row_inds) > 0:
                        _prec = (ul_gt_list[sem_pl_row_inds] == class_ind).float().mean().item()
                    else:
                        _prec = -1
                    sem_prec_dict[str(class_ind)] = _prec

        # done here (recall and precision)
        gt_imb_ratio = float(max(gt_count_dict.values())) / min(gt_count_dict.values())
        self.meters.put_scalar("imb_factors/gt_label", gt_imb_ratio, n=1, show_avg=False)

        if "pseudo_labels" in valid_keys:
            if sum(list(pl_count_dict.values())) == 0:
                pl_imb_ratio = -1
            else:
                pl_counts = [v for v in pl_count_dict.values() if v > 0]
                pl_imb_ratio = float(max(pl_counts)) / min(pl_counts)
            self.meters.put_scalar("imb_factors/main_pl", pl_imb_ratio, n=1, show_avg=False)
            # self.meters.put_scalars(pl_count_dict, show_avg=False, prefix="pl_distribution")

            if not self.is_ul_unknown:
                self.meters.put_scalars(pl_recall_dict, show_avg=False, prefix="main_pl_recall")
                self.meters.put_scalars(pl_prec_dict, show_avg=False, prefix="main_pl_precision")
                self.meters.put_scalars(pl_gain_dict, show_avg=False, prefix="main_pl_gain")
                # recall and precision!

                pl_recall = np.mean(list(pl_recall_dict.values()))
                pl_prec = np.mean(list(pl_prec_dict.values()))
                self.meters.put_scalar("avg_recall/pl", pl_recall, n=1, show_avg=False)
                self.meters.put_scalar("avg_prec/pl", pl_prec, n=1, show_avg=False)

        if with_lin_pl:
            if sum(list(lin_pl_count_dict.values())) == 0:
                lin_pl_imb_ratio = -1
            else:
                _lin_count_list = [float(v) for v in lin_pl_count_dict.values() if v > 0]
                lin_pl_imb_ratio = max(_lin_count_list) / min(_lin_count_list)
            self.meters.put_scalar("imb_factors/linear_pl", lin_pl_imb_ratio, n=1, show_avg=False)

        if with_sem_pl:
            if sum(list(sem_pl_count_dict.values())) == 0:
                sem_pl_imb_ratio = -1
            else:
                _sem_count_list = [float(v) for v in sem_pl_count_dict.values() if v > 0]
                sem_pl_imb_ratio = max(_sem_count_list) / min(_sem_count_list)
            self.meters.put_scalar("imb_factors/semantic_pl", sem_pl_imb_ratio, n=1, show_avg=False)

        if not self.is_ul_unknown:
            if with_lin_pl:
                self.meters.put_scalars(lin_recall_dict, show_avg=False, prefix="linear_pl_recall")
                self.meters.put_scalars(lin_prec_dict, show_avg=False, prefix="linear_pl_precision")
                self.meters.put_scalars(lin_pl_gain_dict, show_avg=False, prefix="linear_pl_gain")

                avg_lin_recall = np.mean(list(lin_recall_dict.values()))
                avg_lin_prec = np.mean(list(lin_prec_dict.values()))
                self.meters.put_scalar("avg_recall/linear_pl", avg_lin_recall, n=1, show_avg=False)
                self.meters.put_scalar("avg_prec/linear_pl", avg_lin_prec, n=1, show_avg=False)

            if with_sem_pl:
                self.meters.put_scalars(sem_recall_dict, show_avg=False, prefix="sem_pl_recall")
                self.meters.put_scalars(sem_prec_dict, show_avg=False, prefix="sem_pl_precision")
                self.meters.put_scalars(sem_pl_gain_dict, show_avg=False, prefix="sem_pl_gain")

                avg_sem_recall = np.mean(list(sem_recall_dict.values()))
                avg_sem_prec = np.mean(list(sem_prec_dict.values()))
                self.meters.put_scalar("avg_recall/sem_pl", avg_sem_recall, n=1, show_avg=False)
                self.meters.put_scalar("avg_prec/sem_pl", avg_sem_prec, n=1, show_avg=False)

        # reset for new bank to accumulate
        self.reset()
