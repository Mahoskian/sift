#!/usr/bin/env python3
# Soham Naik - 05/21/2025
import os
import shutil


def restore_input_folder(output_dir, input_dir):
    if not os.path.exists(output_dir):
        return
    for root, _, files in os.walk(output_dir):
        for f in files:
            os.rename(os.path.join(root, f), os.path.join(input_dir, f))
    shutil.rmtree(output_dir)


def list_input_files(input_dir, mode):
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    VID_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
    exts = VID_EXTS if mode == "video" else IMG_EXTS
    return [
        f
        for f in sorted(os.listdir(input_dir))
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(exts)
    ]


def save_groups(groups, input_dir, output_dir, dry_run):
    os.makedirs(output_dir, exist_ok=True)
    for idx, grp in enumerate(groups, start=1):
        dest = os.path.join(output_dir, f"group_{idx}")
        os.makedirs(dest, exist_ok=True)
        for fname in grp:
            src = os.path.join(input_dir, fname)
            dst = os.path.join(dest, fname)
            if dry_run:
                print(f"DRY_RUN: {src} -> {dst}")
            else:
                os.rename(src, dst)
