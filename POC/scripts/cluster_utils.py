#!/usr/bin/env python3
# Soham Naik - 05/21/2025
import numpy as np
from scipy.spatial.distance import pdist, squareform


def average_hash_from_stack(hash_stack, frame_count):
    return (np.sum(hash_stack, axis=0) > (frame_count // 2)).astype(np.uint8)


def get_frame_indices(total_frames, sample_count):
    return set(np.linspace(0, total_frames - 1, sample_count, dtype=int))


def cluster_hashes(hash_list, threshold):
    names = [n for n, _ in hash_list]
    vecs = np.array([v for _, v in hash_list])
    dist_mat = squareform(pdist(vecs, metric="hamming")) * vecs.shape[1]
    groups = []
    used = set()
    for i, name in enumerate(names):
        if i in used:
            continue
        grp = [name]
        used.add(i)
        for j in range(i + 1, len(names)):
            if j not in used and dist_mat[i, j] <= threshold:
                grp.append(names[j])
                used.add(j)
        if len(grp) > 1:
            groups.append(grp)
    return groups
