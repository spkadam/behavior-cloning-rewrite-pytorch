"""
transforms.py — Custom image transforms for data augmentation.

WHY DATA AUGMENTATION?
----------------------
Deep learning models need LOTS of training data.  Instead of collecting
more, we can artificially expand our dataset by applying random
modifications to existing images:

  • Horizontal flip  — teaches the model that roads are symmetric.
  • Translation      — simulates the car being slightly off-centre.
  • Brightness jitter— simulates different lighting / weather.

In the old Keras code these were raw OpenCV calls scattered through
utils_CUT.py.  In PyTorch the convention is to wrap each augmentation
in a callable class (a "transform") and compose them into a pipeline.

WHAT YOU'LL LEARN:
  • Writing callable classes with __call__.
  • Using NumPy / OpenCV for image manipulation.
  • How transforms can modify BOTH the image AND the label (steering).
  • The Compose pattern for chaining transforms.
"""

import cv2
import numpy as np
from typing import Tuple


# ======================================================================
#  PREPROCESSING (always applied, both train and val)
# ======================================================================

class Resize:
    """Resize image to (width, height)."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __call__(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        image = cv2.resize(image, (self.width, self.height))
        return image, angle


class Crop:
    """
    Crop an image to a region of interest.

    For driving, we remove the sky (top portion) since it carries no
    useful road information and would confuse the model.
    """

    def __init__(self, top: int, bottom: int, left: int, right: int):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __call__(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        image = image[self.top:self.bottom, self.left:self.right, :]
        return image, angle


class BGR2HSV:
    """
    Convert BGR (OpenCV default) to HSV colour space.

    HSV separates colour information (Hue, Saturation) from brightness
    (Value), making it easier for the network to generalise across
    lighting conditions.
    """

    def __call__(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image, angle


class ToFloatTensor:
    """
    Convert a NumPy (H, W, C) uint8 image to a PyTorch (C, H, W) float
    tensor.

    KEY CONCEPT — channel ordering:
      • OpenCV / NumPy store images as (height, width, channels).
      • PyTorch Conv2d expects (channels, height, width).
      We use np.transpose to go from HWC → CHW.

    We keep pixel values in [0, 255] because the model's first operation
    normalises them to [-1, 1].
    """

    def __call__(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        # HWC → CHW  and convert to float32
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return image, angle


# ======================================================================
#  AUGMENTATION (only applied during training)
# ======================================================================

class RandomFlip:
    """
    Randomly flip the image horizontally.

    When we flip the image, we must also negate the steering angle
    (left ↔ right).  This effectively doubles our dataset for free.
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        if np.random.rand() < self.prob:
            image = cv2.flip(image, 1)   # 1 = horizontal flip
            angle = -angle
        return image, angle


class RandomTranslate:
    """
    Randomly shift the image horizontally and vertically.

    A horizontal shift simulates the car being slightly to the left or
    right of centre.  We adjust the steering angle proportionally so the
    model learns the correct recovery behaviour.
    """

    def __init__(
        self,
        range_x: float = 30.0,
        range_y: float = 10.0,
        steer_factor: float = 0.006,
    ):
        self.range_x = range_x
        self.range_y = range_y
        self.steer_factor = steer_factor

    def __call__(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        tx = self.range_x * (np.random.rand() - 0.5)
        ty = self.range_y * (np.random.rand() - 0.5)
        angle += tx * self.steer_factor

        # Build a 2×3 affine transformation matrix for translation
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        h, w = image.shape[:2]
        image = cv2.warpAffine(image, M, (w, h))
        return image, angle


class RandomBrightness:
    """
    Randomly adjust image brightness.

    Converts to HSV, scales the V (value/brightness) channel, then
    converts back.  This makes the model robust to sunny vs cloudy days.
    """

    def __init__(self, max_delta: float = 0.4):
        self.max_delta = max_delta

    def __call__(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        ratio = 1.0 + self.max_delta * (np.random.rand() - 0.5)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return image, angle


class RandomAugment:
    """
    Apply the full augmentation suite with a given probability.

    If the random draw says "no augment", the image passes through
    unchanged — this way the model also sees clean examples.
    """

    def __init__(
        self,
        prob: float = 0.6,
        flip_prob: float = 0.5,
        brightness_range: float = 0.4,
        translate_x: float = 30.0,
        translate_y: float = 10.0,
        steer_factor: float = 0.006,
    ):
        self.prob = prob
        self.flip = RandomFlip(flip_prob)
        self.translate = RandomTranslate(translate_x, translate_y, steer_factor)
        self.brightness = RandomBrightness(brightness_range)

    def __call__(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        if np.random.rand() < self.prob:
            image, angle = self.flip(image, angle)
            image, angle = self.translate(image, angle)
            image, angle = self.brightness(image, angle)
        return image, angle


# ======================================================================
#  COMPOSE — chain multiple transforms
# ======================================================================

class Compose:
    """
    Chain a list of transforms.

    Usage:
        transform = Compose([Resize(640, 480), Crop(180, 480, 10, 640)])
        image, angle = transform(image, angle)

    This is the PyTorch-idiomatic way to build preprocessing pipelines.
    torchvision.transforms.Compose does the same thing, but ours also
    passes the steering angle through.
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        for t in self.transforms:
            image, angle = t(image, angle)
        return image, angle


# ======================================================================
#  FACTORY FUNCTIONS — build standard pipelines
# ======================================================================

def get_train_transforms(cfg) -> Compose:
    """
    Build the full training pipeline:
      augment → resize → crop → BGR→HSV → to tensor
    """
    return Compose([
        RandomAugment(
            prob=cfg.augment_prob,
            flip_prob=cfg.flip_prob,
            brightness_range=cfg.brightness_range,
            translate_x=cfg.translate_range_x,
            translate_y=cfg.translate_range_y,
            steer_factor=cfg.translate_steer_factor,
        ),
        Resize(cfg.raw_width, cfg.raw_height),
        Crop(cfg.crop_top, cfg.crop_bottom, cfg.crop_left, cfg.crop_right),
        BGR2HSV(),
        ToFloatTensor(),
    ])


def get_val_transforms(cfg) -> Compose:
    """
    Validation pipeline — NO augmentation, just preprocessing.
    """
    return Compose([
        Resize(cfg.raw_width, cfg.raw_height),
        Crop(cfg.crop_top, cfg.crop_bottom, cfg.crop_left, cfg.crop_right),
        BGR2HSV(),
        ToFloatTensor(),
    ])
