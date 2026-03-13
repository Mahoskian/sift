# PLAN.md — MediaHashCluster C++ Rewrite

> **Working title**: TBD (candidates: `sift`, `drift`, `flock`, `prism`, `vsift`)
> **Language**: C++17, Linux-first (macOS/Windows later)
> **Build system**: CMake
> **License**: MIT (open source from day one)
> **Philosophy**: fast, minimal, CLI-first. Bellard-inspired — small tools that do one thing absurdly well.

---

## Vision

A high-performance perceptual hashing and clustering engine for visual media.
Given a directory of images (or videos), compress each file's visual identity into
a hash vector, measure pairwise similarity, cluster by proximity, and optionally
visualize the hash space interactively.

**Scope**: images and video. Not arbitrary files — perceptual hashing exploits
spatial/visual structure that non-visual files don't have.

**Architecture**: the CLI does everything (hashing, clustering, grouping, file
operations). The GUI is a lightweight visualization addon — not a replacement.

---

## Core Concepts

### Perceptual Hashing as Dimensional Compression

A perceptual hash compresses the visual identity of a media file into a fixed-size
binary vector (e.g., 16x16 = 256 bits). This vector is a point in N-dimensional
binary space. Unlike cryptographic hashes (where any change = totally different
output), perceptual hashes preserve proximity — similar inputs map to nearby points.

This is what makes the entire pipeline possible:
- **Comparison** = Hamming distance between two points
- **Clustering** = finding dense regions in that space
- **Visualization** = projecting that space down to 2D/3D

### Hash Algorithms

| Algorithm | Technique | Strengths | Cost |
|-----------|-----------|-----------|------|
| **dHash** | Pixel gradient comparison | Fast, near-identical duplicates | Minimal |
| **pHash** | DCT (Discrete Cosine Transform) | Robust to compression, resize, color shift | Moderate |
| **wHash** | Haar wavelet transform | Texture/edge sensitive | Moderate |

All three will be implemented from scratch in C++. The interface is uniform —
swap the algorithm, keep the same pipeline.

### Video Hashing Strategies

Three approaches, in order of implementation priority:

**Option A — Averaged frame hashes** (implement first)
- Sample N evenly-spaced frames from the video
- Hash each frame independently
- Combine via majority voting per bit to produce a single hash
- Simple, fast, works well for visually consistent videos
- Limitation: loses temporal info, averages away scene changes

**Option B — Frame hash set** (implement second, as a flag)
- Sample N frames, hash each, keep the full set as the video's identity
- Compare videos by comparing hash sets (min pairwise distance, % matching frames)
- Preserves scene variation, detects partial overlap between videos
- Tradeoff: more storage, O(N*M) comparison per video pair

**Option C — Scene-level hashing** (stretch goal)
- Detect scene boundaries (large frame-to-frame hash jumps)
- Hash each scene segment separately
- Semantically meaningful — captures the actual structure of the video
- Most complex, scene detection is its own sub-problem

---

## Phased Roadmap

### Phase 1 — Core Hashing Engine

**Goal**: CLI tool that takes an input directory and outputs computed hashes.

- [ ] Project scaffolding: CMake build system, directory structure, CI
- [ ] Image loading (via `stb_image` — single-header, zero-dep)
- [ ] Implement dHash from scratch
- [ ] Implement pHash from scratch (DCT computation)
- [ ] Implement wHash from scratch (Haar wavelet transform)
- [ ] Multi-threaded hashing via thread pool
- [ ] Output format: JSON mapping file paths to hash vectors
- [ ] CLI interface: `mhc hash <dir> --algo=phash --size=16`
- [ ] Benchmarks vs the Python implementation

**Key dependencies**:
- `stb_image` — image decoding (single header, public domain)
- Standard C++17 threading (`std::thread`, `std::mutex`, atomics)

**No external math libraries** — DCT and Haar wavelet are straightforward
enough to implement directly. Small, focused implementations > pulling in FFTW
for a 16x16 transform.

---

### Phase 2 — Distance Computation & Clustering

**Goal**: take computed hashes, cluster them, output groups.

- [ ] Hamming distance computation (SIMD-optimized with `__builtin_popcountll`)
- [ ] Pairwise distance matrix (parallel computation for large sets)
- [ ] Clustering algorithms:
  - [ ] Threshold-based grouping (simple: group if distance < T)
  - [ ] Hierarchical agglomerative clustering
  - [ ] DBSCAN / HDBSCAN (density-based, no hard threshold needed)
- [ ] CLI interface: `mhc cluster --method=threshold --threshold=0.07`
- [ ] File operations: `mhc sort <dir>` (hash + cluster + move files into group folders)
- [ ] Hash caching: save computed hashes to disk, skip re-hashing unchanged files
- [ ] Dry-run mode

**Design note**: the distance matrix is O(n^2) in memory. For very large datasets
(100k+ files), consider approximate nearest neighbor (ANN) approaches or chunked
computation. But optimize later — O(n^2) is fine for the initial target of <50k files.

---

### Phase 3 — Visualization

**Goal**: interactive 2D/3D visualization of the hash space with live parameter tuning.

#### Phase 3a — Python Visualization (first pass)
- [ ] Python script that reads the JSON hash output
- [ ] Dimensionality reduction: PCA (fast), t-SNE, UMAP
- [ ] Interactive scatter plot (Plotly or matplotlib)
- [ ] Points colored by cluster, click to see filename/thumbnail
- [ ] Slider controls: similarity threshold, clustering parameters
- [ ] Live re-clustering on slider change
- [ ] CLI integration: `mhc viz` launches the Python visualizer

#### Phase 3b — Native Visualization (later)
- [ ] Lightweight C++ GUI (Dear ImGui + SDL2 or similar)
- [ ] Real-time rendering of hash space
- [ ] Same interactive controls, but native performance
- [ ] Thumbnail previews of media on hover/click
- [ ] This replaces the Python visualizer as the default

**Visualization philosophy**: this is the least interesting part architecturally.
Keep it lightweight, minimal, and beautiful. The CLI is the real tool — the viz
is a window into the hash space, nothing more.

---

### Phase 4 — Polish & Extend

- [ ] Video support (Option A first, then Option B as `--video-mode=average|set`)
- [ ] Video frame extraction (via FFmpeg libraries or minimal decoder)
- [ ] Scene-level hashing (Option C) as experimental feature
- [ ] Python bindings via pybind11 (use the C++ engine as a Python library)
- [ ] macOS support
- [ ] Windows support
- [ ] Package distribution (Homebrew, AUR, apt, pip for bindings)
- [ ] Export results to HTML report (cluster viz + thumbnails + metadata)
- [ ] Audio perceptual hashing (spectral fingerprinting — natural extension)
- [ ] Benchmarks and comparisons with existing tools
- [ ] Comprehensive documentation and examples

---

## Project Structure (Proposed)

```
├── CMakeLists.txt
├── README.md
├── PLAN.md
├── LICENSE
├── include/
│   ├── hash/
│   │   ├── dhash.h
│   │   ├── phash.h
│   │   └── whash.h
│   ├── cluster/
│   │   ├── distance.h
│   │   ├── threshold.h
│   │   ├── hierarchical.h
│   │   └── dbscan.h
│   ├── io/
│   │   ├── image_loader.h
│   │   ├── video_loader.h
│   │   └── cache.h
│   ├── core/
│   │   ├── threadpool.h
│   │   └── types.h
│   └── cli/
│       └── cli.h
├── src/
│   ├── main.cpp
│   ├── hash/
│   ├── cluster/
│   ├── io/
│   ├── core/
│   └── cli/
├── tests/
├── bench/
├── viz/                  # Python visualization (Phase 3a)
│   ├── visualize.py
│   └── requirements.txt
└── third_party/
    └── stb_image.h
```

---

## Design Principles

1. **CLI does everything** — the GUI is optional. Every operation is scriptable.
2. **Uniform hash interface** — swap algorithms without changing the pipeline.
3. **Speed over generality** — optimize for images/video, don't try to hash everything.
4. **Minimal dependencies** — prefer single-header libraries or writing it yourself.
5. **Progressive complexity** — threshold clustering works out of the box, advanced methods are opt-in.
6. **Composable** — hash output is JSON, can be piped to other tools or consumed by the viz layer.

---

## Open Questions

- [ ] Final project name
- [ ] Hash output format: JSON (human-readable) vs binary (faster I/O for large datasets)? Maybe both.
- [ ] Should the viz layer communicate with the CLI via files, pipes, or sockets?
- [ ] SIMD strategy: SSE4.2 baseline? AVX2 optional? Runtime detection?
- [ ] How to handle hash size configurability at compile time vs runtime (templates vs runtime polymorphism)
