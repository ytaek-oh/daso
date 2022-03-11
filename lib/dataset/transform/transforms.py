import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms
from yacs.config import CfgNode

from typing import Optional, Tuple, Union

from .randaugment import RandAugment


class GeneralizedSSLTransform:

    def __init__(self, transforms: list) -> None:
        assert len(transforms) > 0
        self.transforms = transforms

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> Union[Tensor, Tuple[Tensor]]:
        results = []
        for t in self.transforms:
            results.append(t(img))
        if len(results) == 1:
            return results[0]
        return tuple(results)


class Augmentation:

    def __init__(
        self,
        cfg: CfgNode,
        img_size: Tuple[int],
        *,
        flip: bool = True,
        crop: bool = True,
        strong_aug: bool = False,
        norm_params: Optional[dict] = None,
        is_train: bool = True,
        resolution=32,
        ra_first=False
    ) -> None:
        h, w = img_size
        t = []

        # random horizontal flip
        if flip:
            t.append(transforms.RandomHorizontalFlip())

        # random padding crop
        if crop:
            pad_w = int(w * 0.125) if w == 32 else 4
            pad_h = int(h * 0.125) if h == 32 else 4
            t.append(
                transforms.RandomCrop(img_size, padding=(pad_h, pad_w), padding_mode="reflect")
            )

        if strong_aug and ra_first:
            # apply RA before image resize
            t.append(RandAugment(2, 10, prob=0.5, aug_pool="FixMatch", apply_cutout=True))

        # resize if the actual size of image differs from the desired resolution
        if resolution != h:
            t.append(transforms.Resize((resolution, resolution)))

        if strong_aug and (not ra_first):
            # apply RA after image resize
            t.append(RandAugment(2, 10, prob=0.5, aug_pool="FixMatch", apply_cutout=True))

        # numpy to tensor
        t.append(transforms.ToTensor())

        # normalizer
        if norm_params is not None:
            t.append(transforms.Normalize(**norm_params))

        self.t = transforms.Compose(t)

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> Tensor:
        if isinstance(img, np.ndarray):
            if img.shape[0] == 3:
                img = np.moveaxis(img, 0, -1)
            img = Image.fromarray(img.astype(np.uint8))
        # PIL image type
        assert isinstance(img, Image.Image)
        return self.t(img)
