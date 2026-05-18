#pragma once

#include "hash.hpp"
#include "frame_source.hpp"

#include <filesystem>
#include <functional>

// ─── Video hashing ───────────────────────────────────────────────────────────
//
// hash_video() extracts frames from a video file using the supplied FrameSource,
// hashes each frame with hash_fn, then combines them into a single HashResult
// via majority-vote averaging: each output bit is 1 if more than half the
// frame hashes have that bit set, 0 otherwise.
//
// Returns an empty HashResult (bits empty, size 0) if no frames could be
// extracted (e.g. ffmpeg not installed, unreadable file).

using HashFn = std::function<HashResult(const std::string&, int)>;

HashResult hash_video(
    const std::filesystem::path& video,
    const io::FrameSource& source,
    HashFn hash_fn,
    int hash_size);
