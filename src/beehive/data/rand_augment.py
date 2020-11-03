# From: https://raw.githubusercontent.com/ildoonet/pytorch-randaugment/master/RandAugment/augmentations.py
# -----------------------------------------------------
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np

from PIL import Image


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def ZoomIn(img, v):
    assert 0 <= v <= 0.5
    w, h = img.size
    crop_w = int(v * w)
    crop_h = int(v * h)
    new_w = w - crop_w
    new_h = h - crop_h
    img = PIL.ImageOps.fit(img, size=(new_w,new_h), bleed=v/2, method=Image.BILINEAR)
    return img.resize((w,h), resample=Image.BILINEAR)


def ZoomOut(img, v):
    assert 0 <= v <= 0.5
    w, h = img.size
    pad_w = int(v * w)
    pad_h = int(v * h)
    img = np.asarray(img)
    if img.ndim == 3:
        img = np.pad(img, [(pad_h//2,pad_h//2), (pad_w//2,pad_w//2), (0,0)], mode='reflect')
    elif img.ndim == 2:
        img = np.pad(img, [(pad_h//2,pad_h//2), (pad_w//2,pad_w//2)], mode='reflect')
    img = PIL.Image.fromarray(img)
    return img.resize((w,h), resample=Image.BILINEAR)


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    if random.random() > 0.5:
        return PIL.ImageOps.flip(img)
    else:
        return PIL.ImageOps.mirror(img)


def Downsample(img, _):
    w, h = img.size
    img = img.resize((w//2,h//2), resample=Image.BILINEAR)
    return img.resize((w,h), resample=Image.BILINEAR)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # percentage: [0, 0.2]
    assert 0.0 <= v <= 0.6
    if v <= 0.:
        return img
    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    if v < 0:
        return img
    w, h = img.size
    x0 = random.uniform(0,w)
    y0 = random.uniform(0,h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)
    xy = (x0, y0, x1, y1)
    img = img.copy()
    if img.mode == 'F':
        PIL.ImageDraw.Draw(img).rectangle(xy, np.mean(img))
    else:
        PIL.ImageDraw.Draw(img).rectangle(xy, (127,127,127))
    return img


def Identity(img, v):
    return img


# Spatial transforms only
def augmentations():
    return [
        # (AutoContrast, 0, 1),
        # (Equalize, 0, 1),
        # (Invert, 0, 1),
        (Downsample, 0, 1),
        (ZoomIn, 0, 0.5),
        (ZoomOut, 0, 0.5),
        (Rotate, 0, 30),
        # (Posterize, 0, 4),
        # (Solarize, 0, 256),
        # (SolarizeAdd, 0, 110),
        # (Color, 0.1, 1.9),
        # (Contrast, 0.1, 1.9),
        # (Brightness, 0.1, 1.9),
        # (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (Cutout, 0., 0.6),
        (TranslateX, 0., 0.3),
        (TranslateY, 0., 0.3),
    ]


class RandAugment:

    def __init__(self, n=3, m=12, p=1.0):
        self.n = n
        self.m = m # [0, 30]
        self.p = p
        self.augmentations = augmentations()

    def __call__(self, image):
        if np.random.binomial(1, 1-self.p):
            return {'image': image}
        img = Image.fromarray(image)
        ops = random.choices(self.augmentations, k=self.n)
        for op, minval, maxval in ops:
            m = np.random.poisson(self.m)
            m = 30 if m > 30 else m
            val = (float(m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return {'image': np.asarray(img)}


class RandAugment3d:
    """Each slice is raw HU and must be augmented
    separately in order to work w/ PIL, if using
    contiguous slices.
    """
    def __init__(self, n=3, m=12, p=1.0):
        self.n = n
        self.m = m # [0, 30]
        self.p = p
        self.augmentations = augmentations()

    def __call__(self, image):
        if np.random.binomial(1, 1-self.p):
            return {'image': image}
        imglist = []
        for i in range(image.shape[0]):
            imglist += [Image.fromarray(image[i])]
        ops = random.choices(self.augmentations, k=self.n)
        for op, minval, maxval in ops:
            m = np.random.poisson(self.m)
            m = 30 if m > 30 else m
            val = (float(m) / 30) * float(maxval - minval) + minval
            # For operations that use random.random (Shear, Translate),
            # ensures that each slice gets the same op
            seed = random.random()
            transformed = []
            for img in imglist:
                random.seed(seed)
                transformed += [op(img, val) for img in imglist]
        img = np.concatenate([np.expand_dims(im, axis=0) for im in imglist], axis=0)
        return {'image': np.asarray(img)}



