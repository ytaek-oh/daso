import random

import numpy as np
from torch.utils.data import Sampler
import torch


class ClassAwareSampler(Sampler):

    def __init__(self, data_source, total_samples, beta=0.999, shuffle=True):
        labels = data_source.dataset[data_source.label_key]

        num_classes = len(np.unique(labels))
        label_to_count = [0] * num_classes

        for idx, label in enumerate(labels):
            label_to_count[label] += 1

        if beta < 1:
            effective_num = 1.0 - np.power(beta, label_to_count)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
        else:
            per_cls_weights = 1.0 / np.array(label_to_count)

        weights = torch.DoubleTensor([per_cls_weights[label] for label in labels])

        # total train epochs
        num_epochs = int(total_samples / len(labels)) + 1
        total_inds = []
        for epoch in range(num_epochs):
            inds_list = torch.multinomial(weights, len(labels), replacement=True).tolist()
            if shuffle:
                random.shuffle(inds_list)
            total_inds.extend(inds_list)
        total_inds = total_inds[:total_samples]

        self.per_cls_prob = per_cls_weights / np.sum(per_cls_weights)

        self._indices = total_inds

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)
