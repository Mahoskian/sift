#!/usr/bin/env python3
# Soham Naik - 05/21/2025
import os
import time
import argparse
from tqdm.contrib.concurrent import process_map

# Hashing algorithms
from hash_algorithms import dhash_compute, phash_compute, whash_compute

# I/O helpers
from io_utils import restore_input_folder, list_input_files, save_groups

# Hash‐and‐frame‐sampling wrappers
from hash_utils import set_compute_fn, worker

# Clustering routines
from cluster_utils import cluster_hashes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Group visually similar media using dHash, pHash, or wHash."
    )
    parser.add_argument(
        "--hash-type",
        choices=["dhash", "phash", "whash"],
        required=True,
        help="Select hashing algorithm.",
    )
    parser.add_argument(
        "--mode",
        choices=["image", "video"],
        default="image",
        help="Processing mode: image or video.",
    )
    parser.add_argument(
        "--input",
        dest="input_dir",
        default="input",
        help="Directory containing media files.",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default="output",
        help="Directory for grouped output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count(), 10),
        help="Max parallel hashing processes.",
    )
    parser.add_argument(
        "--chunk-coef",
        type=int,
        default=4,
        help="Batch coefficient for chunksize calculation.",
    )
    parser.add_argument(
        "--frames-to-sample",
        type=int,
        default=20,
        help="Number of frames to sample per video.",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=None,
        help="Hash grid size override (uses algorithm default if omitted).",
    )
    parser.add_argument(
        "--threshold",
        dest="similarity_threshold",
        type=int,
        default=None,
        help="Similarity threshold override (uses algorithm default if omitted).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate file moves without executing."
    )
    parser.add_argument(
        "--graph", action="store_true", help="Plot clusters in 2D after grouping."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Choose algorithm and defaults
    if args.hash_type == "dhash":
        compute_fn, default_size, default_thresh = dhash_compute, 8, 10
    elif args.hash_type == "phash":
        compute_fn, default_size, default_thresh = phash_compute, 16, 25
    else:
        compute_fn, default_size, default_thresh = whash_compute, 8, 10

    # Tell hash_utils which function to use
    set_compute_fn(compute_fn)

    # Apply overrides or defaults
    hash_size = args.hash_size or default_size
    thresh = args.similarity_threshold or default_thresh

    start_time = time.time()
    print(
        f"Using {args.workers} workers, hash={args.hash_type}, size={hash_size}, thresh={thresh}"
    )

    # Prepare workspace
    restore_input_folder(args.output_dir, args.input_dir)
    files = list_input_files(args.input_dir, args.mode)
    print(f"Found {len(files)} {args.mode} files to process")

    # Chunking for multiprocessing
    chunk_size = max(1, len(files) // (args.workers * args.chunk_coef))
    tasks = zip(
        files,
        [args.input_dir] * len(files),
        [args.mode] * len(files),
        [hash_size] * len(files),
        [args.frames_to_sample] * len(files),
    )

    results = process_map(
        worker,
        tasks,
        max_workers=args.workers,
        desc="Hashing files",
        chunksize=chunk_size,
    )

    valid = [r for r in results if r]
    groups = cluster_hashes(valid, thresh)

    save_groups(groups, args.input_dir, args.output_dir, args.dry_run)

    print(f"Identified {len(groups)} groups")
    print(f"Completed in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
