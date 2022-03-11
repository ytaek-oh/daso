import os
import shutil

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    *,
    is_best: bool,
    ema_model: nn.Module = None,
    meta_dict: dict = None,
    file_name: str = "checkpoint.pth.tar"
) -> None:
    state_dict = {
        "model": get_model_state_dict(model),
        "ema_model": get_model_state_dict(ema_model) if ema_model is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "meta": meta_dict
    }
    file_path = os.path.join(save_path, file_name)
    torch.save(state_dict, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(save_path, "model_best.pth.tar"))


def get_model_state_dict(model: nn.Module) -> dict:
    """pill out `nn.Module` not to have "module" attribute"""
    model = model.module if hasattr(model, "module") else model
    return model.state_dict()
