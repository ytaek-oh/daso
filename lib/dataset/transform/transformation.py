import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


def _float_parameter(level: int, max_level: float, *, magnitude: int = 10):
    """
        args:
            level (int): [1, magnitude]

        returns:
            level (float): in range of [max_level / magnitude, max_level]
    """
    return float(level) / magnitude * max_level


def _int_parameter(level: int, max_level: int, *, magnitude: int = 10):
    """
        args:
            level (int): [1, magnitude]

        returns:
            level (int): [max_level / magnitude, max_level]
    """
    return int(level / magnitude * max_level)


def AutoContrast(img: Image.Image, *args, **kwargs) -> Image:
    return ImageOps.autocontrast(img)


def Brightness(img: Image.Image, level: int, max_level: float, *, bias: float = 0) -> Image:
    level = _float_parameter(level, max_level) + bias
    return ImageEnhance.Brightness(img).enhance(level)


def Color(img: Image.Image, level: int, max_level: float, *, bias: float = 0) -> Image:
    level = _float_parameter(level, max_level) + bias
    return ImageEnhance.Color(img).enhance(level)


def Contrast(img: Image.Image, level: int, max_level: float, *, bias: float = 0) -> Image:
    level = _float_parameter(level, max_level) + bias
    return ImageEnhance.Contrast(img).enhance(level)


def Equalize(img: Image.Image, *args, **kwargs) -> Image:
    return ImageOps.equalize(img)


def Identity(img: Image.Image, *args, **kwargs) -> Image:
    return img


def Posterize(img: Image.Image, level: int, max_level: int, *, bias: int = 0) -> Image:
    level = _int_parameter(level, max_level) + bias
    return ImageOps.posterize(img, level)


def Rotate(img: Image.Image, level: int, max_level: int, *, bias: int = 0) -> Image:
    level = _int_parameter(level, max_level) + bias
    if random.random() < 0.5:
        level = -level
    return img.rotate(level)


def Sharpness(img: Image.Image, level: int, max_level: float, *, bias: float = 0) -> Image:
    level = _float_parameter(level, max_level) + bias
    return ImageEnhance.Sharpness(img).enhance(level)


def ShearX(img: Image.Image, level: int, max_level: float, *, bias: float = 0) -> Image:
    level = _float_parameter(level, max_level) + bias
    if random.random() < 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))


def ShearY(img: Image.Image, level: int, max_level: float, *, bias: float = 0) -> Image:
    level = _float_parameter(level, max_level) + bias
    if random.random() < 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))


def Solarize(img: Image.Image, level: int, max_level: int, *, bias: int = 0) -> Image:
    v = _int_parameter(level, max_level) + bias
    return ImageOps.solarize(img, 256 - v)


def TranslateX(img: Image.Image, level: int, max_level: float, *, bias: float = 0) -> Image:
    level = _float_parameter(level, max_level) + bias
    if random.random() < 0.5:
        level = -level
    level = int(level * img.size[0])
    return img.transform(img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))


def TranslateY(img: Image.Image, level: int, max_level: float, *, bias: float = 0) -> Image:
    level = _float_parameter(level, max_level) + bias
    if random.random() < 0.5:
        level = -level
    level = int(level * img.size[1])
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))


def CutoutAbs(img: Image.Image, level: int, *args, **kwargs):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - level / 2.))
    y0 = int(max(0, y0 - level / 2.))
    x1 = int(min(w, x0 + level))
    y1 = int(min(h, y0 + level))
    xy = (x0, y0, x1, y1)

    # gray
    color = (127, 127, 127)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


aug_pools = {
    "FixMatch": [
        # (Op, max_level, bias)
        (AutoContrast, None, None),
        (Brightness, 0.9, 0.05),
        (Color, 0.9, 0.05),
        (Contrast, 0.9, 0.05),
        (Equalize, None, None),
        (Identity, None, None),
        (Posterize, 4, 4),
        (Rotate, 30, 0),
        (Sharpness, 0.9, 0.05),
        (ShearX, 0.3, 0),
        (ShearY, 0.3, 0),
        (Solarize, 256, 0),
        (TranslateX, 0.3, 0),
        (TranslateY, 0.3, 0)
    ]
}
