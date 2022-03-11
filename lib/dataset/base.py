import numpy as np
from torch.utils.data import Dataset


class BaseNumpyDataset(Dataset):
    """Custom dataset class for classification"""

    def __init__(
        self,
        data_dict: dict,
        image_key: str = "images",
        label_key: str = "labels",
        transforms=None,
        is_ul_unknown=False
    ):
        self.dataset = data_dict
        self.image_key = image_key
        self.label_key = label_key
        self.transforms = transforms
        self.is_ul_unknown = is_ul_unknown

        if not is_ul_unknown:
            self.num_samples_per_class = self._load_num_samples_per_class()
        else:
            self.num_samples_per_class = None

    def __getitem__(self, idx):
        img = self.dataset[self.image_key][idx]
        label = -1 if self.is_ul_unknown else self.dataset[self.label_key][idx]
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label, idx

    def __len__(self):
        return len(self.dataset[self.image_key])

    # label-to-class quantity
    def _load_num_samples_per_class(self):
        labels = self.dataset[self.label_key]
        num_classes = len(np.unique(labels))

        classwise_num_samples = dict()
        for i in range(num_classes):
            classwise_num_samples[i] = len(np.where(labels == i)[0])

        # in a descending order of classwise count. [(class_idx, count), ...]
        res = sorted(classwise_num_samples.items(), key=(lambda x: x[1]), reverse=True)
        return res

    def select_dataset(self, indices=None, labels=None, return_transforms=False):
        if indices is None:
            indices = list(range(len(self)))

        imgs = self.dataset[self.image_key][indices]

        if not self.is_ul_unknown:
            _labels = self.dataset[self.label_key][indices]
        else:
            _labels = np.array([-1 for _ in range(len(indices))])

        if labels is not None:
            # override specified labels (i.e., pseudo-labels)
            _labels = np.array(labels)

        assert len(_labels) == len(imgs)
        dataset = {self.image_key: imgs, self.label_key: _labels}

        if return_transforms:
            return dataset, self.transforms    
        return dataset
