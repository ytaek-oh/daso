import datetime
import json
import logging

import torch

from .meters import Meters


class Writer:
    """
    Base class for writers that obtain events from :class:`Meters` and process them.
    """

    def write(self):
        raise NotImplementedError

    def close(self):
        pass


class CommonMetricPrinter(Writer):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    """

    def __init__(self, max_iter: int) -> None:
        self.logger = logging.getLogger(__name__)
        self._max_iter = max_iter

    def write(self, meters: Meters) -> None:
        metrics = meters.get_latest_scalars_with_avg()
        iteration = meters.iter

        data_time, time = None, None
        eta_string = "N/A"
        try:
            data_time = metrics["misc/data_time"]
            time = metrics["misc/iter_time"]
            eta_seconds = time * (self._max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:  # they may not exist in the first few iterations (due to warmup)
            pass
        try:
            lr = "{:.6f}".format(metrics["misc/lr"])
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        losses_str = "  ".join([f"{k}: {v:.5f}" for k, v in metrics.items() if "loss" in k])
        data_time_print = f"data_time: {1.0 / data_time:.1f}"
        self.logger.info(
            " eta: {eta}  {iter}  {losses}  {time}  {data_time}  lr: {lr}  {memory}".format(
                eta=eta_string,
                iter="iteration: {}/{}".format(iteration, self._max_iter),
                losses=losses_str,
                time="iter_time: {:.4f}".format(time) if time is not None else "",
                data_time=data_time_print if data_time is not None else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )

        """
        # evaluated
        if "test/top1" in metrics.keys():
            test_top1 = metrics["test/top1"]
            log_text = "Evaluation results: test_top1: {:.1f}".format(test_top1)
            if "test/top1_median20" in metrics.keys():
                test_top1_med20 = metrics["test/top1_median20"]
                log_text += "  test_top1_median20: {:.1f}".format(test_top1_med20)
            self.logger.info(log_text)
        """

        log_text = "Evaluation results: "
        prefix_keys = ["test/top1", "test/top1_median20", "test/top1_la", "test/top1_la_median20"]
        for _prefix in prefix_keys:
            if _prefix in metrics.keys():
                log_text += "{}: {:.1f}  ".format(_prefix, metrics[_prefix])
        self.logger.info(log_text)


class JSONWriter(Writer):
    """
    Write scalars to a json file.

    It saves scalars as one json per line (instead of a big json) for easy parsing.

    Examples parsing such a json file:
    ::
        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 20,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 40,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]

        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...

    """

    def __init__(self, json_file):
        """
        Args:
            json_file (str): path to the json file. New data will be appended if the file exists.
        """
        self.json_file = json_file
        self._file_handle = open(self.json_file, "a")

    def write(self, meters):
        metrics = meters.get_latest_scalars_with_avg()
        metrics["iteration"] = meters.iter

        # _file_handle = open(self.json_file, "a")
        self._file_handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        # _file_handle.close()

    def close(self):
        self._file_handle.close()


class TensorboardWriter(Writer):

    def __init__(self, log_dir: str, **kwargs) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir, **kwargs)

    def write(self, meters: Meters) -> None:
        metrics = meters.get_latest_scalars_with_avg()
        for k, v in metrics.items():
            self._writer.add_scalar(k, v, meters.iter)

    def close(self) -> None:
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()
