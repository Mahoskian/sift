#!/usr/bin/env python3
# Soham Naik - 05/21/2025
import os
import cv2
import numpy as np
import av

from cluster_utils import get_frame_indices, average_hash_from_stack

# Module‑scoped compute function (set by CLI)
_compute_fn = None


def set_compute_fn(fn):
    """Set the hash-compute function (dhash, phash, or whash)."""
    global _compute_fn
    _compute_fn = fn


def hash_image(filename, input_dir, hash_size):
    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return filename, _compute_fn(gray, hash_size)


def hash_video(filename, input_dir, hash_size, frames_to_sample):
    path = os.path.join(input_dir, filename)
    try:
        container = av.open(path)
        stream = container.streams.video[0]
        total = stream.frames or sum(1 for _ in container.decode(stream))
        indices = get_frame_indices(total, frames_to_sample)
        hashes = []
        for idx, frame in enumerate(container.decode(stream)):
            if idx in indices:
                arr = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                hashes.append(_compute_fn(gray, hash_size))
                if len(hashes) >= frames_to_sample:
                    break
        container.close()
        if not hashes:
            return None
        return filename, average_hash_from_stack(np.stack(hashes), len(hashes))
    except Exception:
        return None


def worker(task):
    """Top-level worker for multiprocessing (picklable)."""
    name, inp, mode, sz, fr = task
    if mode == "image":
        return hash_image(name, inp, sz)
    return hash_video(name, inp, sz, fr)
