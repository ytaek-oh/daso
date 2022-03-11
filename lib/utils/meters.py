from collections import defaultdict

import numpy as np


def get_last_n_median(num_list: list, n=20) -> float:
    return np.median(np.asarray(num_list)[-n:])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, *, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Meters:
    """Class including a set of AverageMeter"""

    def __init__(self, start_iter: int = 0) -> None:
        self.meters = defaultdict(AverageMeter)
        self._show_avg = {}
        self._iter = start_iter

    def put_scalar(self, name: str, val: float, *, n: int = 1, show_avg: bool = True) -> None:
        self.meters[name].update(val=val, n=n)
        show_average = self._show_avg.get(name)
        if show_average is not None:
            assert show_average == show_avg
        else:
            self._show_avg[name] = show_avg

    def put_scalars(
        self, scalars_dict: dict, *, n: int = 1, show_avg: bool = True, prefix: str = ""
    ) -> None:
        if prefix:
            prefix = prefix + "/"
        for k, v in scalars_dict.items():
            self.put_scalar(name=prefix + k, val=v, show_avg=show_avg, n=n)

    def reset(self) -> None:
        self.meters = defaultdict(AverageMeter)

    def get_latest_scalars_with_avg(self) -> dict:
        result = {}
        for k, meter in self.meters.items():
            result[k] = meter.avg if self._show_avg[k] else meter.val
        return result

    def step(self) -> None:
        self._iter += 1

    @property
    def iter(self) -> int:
        return self._iter
