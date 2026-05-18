#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace io {

// ─── FrameSource ────────────────────────────────────────────────────────────
//
// Abstract base for frame extraction strategies.
//
// To add a new strategy (scene-change detection, explicit timestamps, etc.):
//   1. Subclass FrameSource and implement extract() + strategy_name().
//   2. Register it in make_frame_source().
//   3. No other code changes required — hash_video() is strategy-agnostic.
//
// All implementations write frames as PNG files into a caller-provided
// temporary directory.  The caller owns the directory and its cleanup.

class FrameSource {
public:
    virtual ~FrameSource() = default;

    // Extract frames from `video` into `out_dir`.
    // Returns paths to the extracted PNG files (may be fewer than requested
    // if the video is short or ffmpeg fails on individual seeks).
    // Returns an empty vector if ffmpeg is unavailable or the video cannot
    // be opened.
    virtual std::vector<std::filesystem::path> extract(
        const std::filesystem::path& video,
        const std::filesystem::path& out_dir) const = 0;

    virtual std::string strategy_name() const = 0;
};


// ─── EvenlySpacedSource ──────────────────────────────────────────────────────
//
// Extracts `n_frames` frames at evenly-spaced timestamps across the video.
// Uses ffprobe to determine duration, then ffmpeg for each seek + extract.

class EvenlySpacedSource final : public FrameSource {
public:
    explicit EvenlySpacedSource(int n_frames = 8);

    std::vector<std::filesystem::path> extract(
        const std::filesystem::path& video,
        const std::filesystem::path& out_dir) const override;

    std::string strategy_name() const override { return "evenly_spaced"; }

private:
    int n_frames_;

    // Query video duration in seconds via ffprobe.
    // Returns 0.0 if ffprobe is unavailable or the file cannot be read.
    double probe_duration(const std::filesystem::path& video) const;
};


// ─── Factory ─────────────────────────────────────────────────────────────────

// Return a FrameSource for the given strategy name.
// Recognised names: "evenly_spaced" (default for any unknown name).
std::unique_ptr<FrameSource> make_frame_source(const std::string& strategy,
                                               int n_frames);

} // namespace io
