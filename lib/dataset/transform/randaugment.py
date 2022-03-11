import random

import numpy as np
from PIL import Image

from .transformation import CutoutAbs, aug_pools


class RandAugment:
    """RandAugment class"""

    def __init__(
        self,
        num_ops: int,
        magnitude: int,
        *,
        prob: float = 0.5,
        aug_pool: str = "FixMatch",
        apply_cutout: bool = True
    ) -> None:
        """
            args:
                num_ops (int)
                magnitude (int)
                prob (float)
        """
        assert num_ops > 1
        self.num_ops = num_ops

        assert magnitude >= 1 and magnitude <= 10
        self.magnitude = magnitude

        self.prob = prob
        self.aug_pool = aug_pools[aug_pool]
        self.apply_cutout = apply_cutout

    def __call__(self, img: Image.Image) -> Image:
        """Apply augmentations to PIL image"""
        ops = random.choices(self.aug_pool, k=self.num_ops)
        for (op, max_level, bias) in ops:
            if random.random() <= 0.5:
                level = np.random.randint(1, self.magnitude)
                img = op(img, level, max_level, bias=bias)

        if self.apply_cutout:
            img = CutoutAbs(img, 16)

        return img

    def __repr__(self) -> str:
        return f"RandAugment(num_ops={self.num_ops}, magnitude={self.magnitude})"
