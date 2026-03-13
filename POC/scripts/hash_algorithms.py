#!/usr/bin/env python3
# Soham Naik - 05/17/2025
import cv2
import numpy as np
from PIL import Image
import imagehash


def dhash_compute(image_gray: np.ndarray, hash_size: int) -> np.ndarray:
    # Difference hash on a grayscale array.
    resized = cv2.resize(
        image_gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA
    )
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.ravel().astype(np.uint8)


def phash_compute(image_gray: np.ndarray, hash_size: int) -> np.ndarray:
    # Perceptual (DCT) hash on a grayscale or RGB array.
    if image_gray.ndim == 3:
        img = Image.fromarray(cv2.cvtColor(image_gray, cv2.COLOR_BGR2RGB))
    else:
        img = Image.fromarray(image_gray)
    ph = imagehash.phash(img, hash_size=hash_size)
    return ph.hash.flatten().astype(np.uint8)


def whash_compute(image_gray: np.ndarray, hash_size: int) -> np.ndarray:
    # Wavelet hash on a grayscale or RGB array.
    if image_gray.ndim == 3:
        img = Image.fromarray(cv2.cvtColor(image_gray, cv2.COLOR_BGR2RGB))
    else:
        img = Image.fromarray(image_gray)
    wh = imagehash.whash(img, hash_size=hash_size)
    return wh.hash.flatten().astype(np.uint8)
