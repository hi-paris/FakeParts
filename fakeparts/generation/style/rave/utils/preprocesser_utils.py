"""Depth preprocessing used by the RAVE adapter."""

import cv2
import numpy as np

from annotator.midas import apply_midas


def _pad64(size):
    return int(np.ceil(float(size) / 64.0) * 64 - size)


def _resize_with_pad(image, resolution):
    assert image.dtype == np.uint8 and image.ndim == 3
    height, width, channels = image.shape
    assert channels in (1, 3, 4)
    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    elif channels == 4:
        color = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        image = (color * alpha + 255.0 * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    scale = float(resolution) / min(height, width)
    target_height = int(np.round(height * scale))
    target_width = int(np.round(width * scale))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    padded = np.pad(
        image,
        [[0, _pad64(target_height)], [0, _pad64(target_width)], [0, 0]],
        mode="edge",
    )

    def remove_pad(value):
        return np.ascontiguousarray(value[:target_height, :target_width].copy())

    return np.ascontiguousarray(padded), remove_pad


def _depth_midas(image, resolution):
    image, remove_pad = _resize_with_pad(image, resolution)
    result, _ = apply_midas(image, np.pi * 2.0)
    return remove_pad(result)


def pixel_perfect_process(input_image, preprocessor_name):
    if preprocessor_name != "depth_midas":
        raise ValueError(f"Unsupported RAVE preprocessor: {preprocessor_name}")
    return _depth_midas(input_image, input_image.shape[0])
