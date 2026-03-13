**Planned Improvements**

1. **Refactor Core Logic**

   * Move shared logic from `phash`, `dhash`, and `whash` scripts into a common `core.py` module.
   * Keep hash-specific behavior and defaults within their respective scripts.

2. **Improve `README.md`**

   * Clarify how to use the tool and what it's for.
   * Explain the differences between dHash, pHash, and wHash: when to use each.
   * Include real-world use cases and example command-line usage.

3. **Add 2D Cluster Visualization**

   * Project high-dimensional hash vectors into 2D space (e.g. using t-SNE or UMAP).
   * Color points by detected group.
   * Support zooming and panning.
   * Allow clicking on a point to view the media filename or open the file.

4. **Build a Proper CLI Interface**

   * Wrap the tool into a unified command-line interface with clear subcommands and flags.
   * Make it pip-installable with an entry point.
   * Optional: add GUI later for user convenience.

5. **Implement Hash Caching**

   * Save media paths & computed hashes [KEY:VALUE] to `.npy` or `.json`.
   * Skip re-hashing unchanged files on rerun.
   * Add option to clear or inspect cache (e.g. size, entry count).

6. **Export Results to HTML**
   * Generate a standalone HTML report showing:
     * Cluster visualization
     * Group summaries
     * Media thumbnails and file info
