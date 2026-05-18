# SIFT — Design Notes

This document captures the architectural thinking, open directions, and unfinished work for the project. It's a living document — update it as decisions get made or ideas get validated.

---

## What exists today

### CLI (`src/`)

- Three hash algorithms: dHash (gradient), pHash (DCT), wHash (wavelet)
- Video hashing: evenly-spaced frame extraction via ffmpeg + majority-vote averaging across N frames
- Three clustering algorithms: threshold, hierarchical agglomerative, HDBSCAN
- Two projection algorithms: PCA, t-SNE
- Parallel hashing via a work-stealing thread pool
- Real-time progress emission to stderr (`sift: progress N/TOTAL`) so the viz can show a live progress bar
- Fully composable JSON pipeline — every stage reads/writes plain JSON and can be piped

### Visualizer (`viz/`)

- Interactive scatter plot in 2D and 3D (matplotlib + tkinter)
- Pre-computed 2D and 3D projections upfront — switching views is instant
- Live re-clustering on slider change (debounced, runs `sift cluster` in background)
- Point click to select and preview media
- Group navigation (◀ ▶ through cluster members), starting from the clicked point
- Inline video playback: ffmpeg pipe → raw RGB frames → tkinter canvas at native FPS
- Cancel button during hashing pipeline
- Accurate progress bar during hashing; indeterminate during projection and clustering
- Media type toggle: images / videos / both
- Frames-per-video slider (controls how many frames are sampled for video hashing)

---

## What is unfinished

### Video frame strategies

The `FrameSource` abstraction exists and is designed to be extended — `extract(video, out_dir)` is the only interface a new strategy needs to implement. Currently only `EvenlySpacedSource` exists. Two obvious next strategies:

- **Scene-change detection**: detect large frame-to-frame hash jumps, extract one representative frame per scene. Would give a much more semantically meaningful fingerprint for videos with multiple scenes. The scene boundary detection itself can be done by hashing consecutive frames and flagging large Hamming jumps — no external tool needed.
- **Explicit timestamps**: let the user specify `--timestamps=0:30,1:15,2:00` to extract frames at meaningful moments. Useful when the user knows the structure of their videos (e.g., always hash the thumbnail frame).

### Hash caching

Every run re-hashes every file from scratch. For large libraries this is slow. The fix is a cache file (e.g. `~/.cache/sift/hashes.db` or a project-local `.sift-cache`) keyed by file path + mtime + size. Only re-hash files that changed. This is the single biggest quality-of-life improvement for regular use on a stable library.

### Weighted multi-hash ensemble

The viz currently lets the user pick one hash algorithm. Ideally the user could combine all three with tunable weights:

```
D(a,b) = w₁·hamming(a.dhash, b.dhash)
        + w₂·hamming(a.phash, b.phash)
        + w₃·hamming(a.whash, b.whash)
```

This requires:
- The CLI to compute and output all three hashes per file simultaneously
- The distance computation to accept a weight vector
- The viz to expose three weight sliders (summing to 1.0) and re-cluster live

The architecture is ready for this — `HashResult` would become a `MultiHashResult` with three members, and `compute_distance_matrix` takes a weight triple.

---

## Directions we haven't explored yet

### Richer descriptors

Perceptual hashes capture structure and frequency but miss two important visual properties: **colour** and **motion**. Adding either would dramatically improve clustering quality for video collections:

**Colour histogram descriptor**
- Divide each frame into a grid (e.g. 4×4 regions)
- Compute the average HSV values per region
- Concatenate into a ~48-float descriptor
- Normalise and combine with the hash-based distance matrix
- This alone would separate "red jacket scenes" from "blue shirt scenes" which all three current hashes treat as equivalent

**Optical flow summary**
- Use ffmpeg's `mestimate` filter or similar to compute per-frame motion vectors
- Summarise as average magnitude + dominant direction across a grid
- A ~32-float descriptor that distinguishes "static talking head" from "person walking" from "action scene"
- Unique to video — no image equivalent, so purely additive

**HOG (Histogram of Oriented Gradients)**
- More discriminative than dHash for edge patterns
- Captures oriented edge density across a grid of cells
- ~256-float descriptor, cheap to compute
- Bridges the gap between pixel-level hashes and semantic descriptors

### Semantic embeddings (CLIP)

This is the high-end option that changes the quality of clustering fundamentally. CLIP (or similar vision-language models) produces a 512-dim float vector where geometric distance corresponds to semantic similarity:

- "Person at a desk" and "person in a boardroom" would cluster together
- "Person outdoors summer" and "person outdoors winter" would be near each other but separate from indoor scenes
- Clothing colour, scene type, activity — all captured automatically

The practical constraints:
- Model weight download (~400MB for ViT-B/32)
- ~200ms per image on CPU, ~20ms on GPU — still usable for libraries up to a few thousand files
- Adds a significant Python dependency (`transformers` or `open_clip`)
- Should be implemented as an optional descriptor — if model not present, falls back to hash-only

The architectural question is where this lives. Options:
1. A Python script that adds CLIP embeddings to the hash JSON (cleanest separation)
2. A new `sift embed` subcommand that calls a Python subprocess internally
3. A separate `sift-embed` binary with its own dependency chain

Option 1 is simplest: `python embed.py hashes.json --output=hashes_with_embeddings.json` and the downstream tools accept either format.

### Descriptor fusion architecture

When multiple descriptors exist (hashes + colour + optical flow + CLIP), the cleanest architecture is:

1. Each descriptor produces a normalised N×N distance matrix
2. The matrices are combined with tunable weights: `D = Σ wᵢ · Dᵢ`
3. The visualizer exposes one weight slider per active descriptor
4. Live re-clustering runs on the combined matrix

This is the general form of what we have now — the current pipeline is a special case with one descriptor and weight 1.0.

The combined distance matrix is the right abstraction: it's descriptor-agnostic, lets you mix binary Hamming distances with float L2 distances (after normalisation), and the clustering algorithms don't need to change at all.

### What we observed about the geometry

When clustering videos of the same person in different scenes, PCA projected everything onto a single axis (or a star pattern with a few arms). This is correct and informative:

- The hash space is a binary hypercube `{0,1}^64`
- When most files are of the same subject, most bits are identical — only a handful vary
- PCA finds that one direction explains nearly all the variance
- The "axis" encodes the dominant mode of variation (probably: scene brightness, or background complexity, or clothing hue)
- The files at the two ends of the axis differ most in that one property

t-SNE pulls clusters apart regardless of global structure, which is why it looks more "spread out" — but both representations are true. PCA says "these are all very similar"; t-SNE says "but these subgroups are distinguishable." Adding a colour histogram descriptor would break the PCA axis into multiple meaningful dimensions.

### Alternative projection methods

Only PCA and t-SNE are implemented. Two others worth considering:

**UMAP** (Uniform Manifold Approximation and Projection)
- Often better than t-SNE: faster, more stable across runs, and unlike t-SNE the distances between clusters are somewhat meaningful
- Python library available (`umap-learn`), would slot into the existing projection pipeline easily
- Recommended as the next projection method to add

**MDS** (Multidimensional Scaling)
- Tries to preserve all pairwise distances, not just local neighbourhood
- Slower than PCA for large N, but the layout is more globally honest than t-SNE
- Useful for small-to-medium collections where global structure matters

### File operations

The CLI currently only outputs JSON — it doesn't touch your files. Useful operations to add:

- `sift sort <dir>` — hash + cluster + move files into group subdirectories (`group_1/`, `group_2/`, `ungrouped/`)
- `sift dedup <dir>` — find and report (or delete) near-duplicate files
- `sift diff <dir-a> <dir-b>` — find files in A that have no close match in B (useful for finding new content)
- `--dry-run` flag on all file operations

### Export and reporting

- HTML export: static page with the scatter plot, thumbnails per cluster, metadata table
- CSV export of the distance matrix for use in other tools
- Integration with media managers (Plex metadata, Darktable/Lightroom-style tagging)

### Performance

- **SIMD Hamming distance**: `__builtin_popcountll` on 64-bit words is already fast, but explicit AVX2 vectorisation would be faster for large N
- **Approximate nearest neighbours**: for N > 50k, O(N²) distance matrix becomes the bottleneck. Libraries like FAISS (Facebook AI Similarity Search) support approximate Hamming distance at scale
- **Streaming hashing**: hash files as they are discovered rather than scanning the whole directory first — reduces peak memory and gives earlier progress feedback
- **Incremental re-cluster**: when only the threshold slider changes, recompute clusters from the cached distance matrix without re-running the full pipeline

### Platform support

Currently Linux-only (uses `xdg-open`, POSIX temp dirs, etc.). The code is mostly portable — the gaps are:
- `getpid()` in video_hash.cpp — use `std::filesystem::temp_directory_path()` with a UUID instead
- `xdg-open` in the viz — already handled per-platform in `system_open()`
- The Makefile assumes Unix conventions

macOS should be straightforward. Windows would need a few more shims.

---

## Decisions made and why

| Decision | Reasoning |
|---|---|
| JSON as the interchange format | Human-readable, pipeable, no schema needed, easy to extend |
| ffmpeg as a subprocess rather than a linked library | Zero link-time dependency, no license complications, version-agnostic |
| stb_image for image decoding | Single-header, public domain, no build system changes |
| Pre-compute both 2D and 3D projections | Switching views is instant; projection is fast enough that doubling it is cheap |
| Majority-vote frame averaging for video | Simple, fast, works well for visually consistent videos; extensible via FrameSource |
| HDBSCAN as the recommended clustering method | No hard threshold needed, handles noise, clusters of arbitrary shape, best quality for real media libraries |
| Hamming distance over other metrics | Natural for binary vectors, O(1) with popcount, exact rather than approximate |
