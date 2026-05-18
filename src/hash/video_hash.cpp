#include "video_hash.hpp"

#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

HashResult hash_video(
    const fs::path& video,
    const io::FrameSource& source,
    HashFn hash_fn,
    int hash_size)
{
    // Create a unique temp directory for this video's frames.
    // Name it after the video's filename + pid to avoid collisions.
    fs::path tmp_dir = fs::temp_directory_path()
        / ("sift_frames_" + video.filename().string()
           + "_" + std::to_string(getpid()));

    fs::create_directories(tmp_dir);

    std::vector<fs::path> frame_paths = source.extract(video, tmp_dir);

    // Clean up temp frames on return (RAII-style via lambda at end).
    auto cleanup = [&]() {
        std::error_code ec;
        fs::remove_all(tmp_dir, ec);
    };

    if (frame_paths.empty()) {
        cleanup();
        return {};
    }

    int n_bits = hash_size * hash_size;
    int n_bytes = (n_bits + 7) / 8;

    // Accumulate per-bit vote counts.
    std::vector<int> votes(n_bits, 0);
    int valid_frames = 0;

    for (const auto& frame : frame_paths) {
        HashResult h = hash_fn(frame.string(), hash_size);
        if (h.bits.empty()) continue;

        for (int b = 0; b < n_bits; ++b) {
            if (h.get_bit(b)) votes[b]++;
        }
        valid_frames++;
    }

    cleanup();

    if (valid_frames == 0) return {};

    // Majority vote: bit is 1 if strictly more than half the frames voted 1.
    HashResult result;
    result.size = hash_size;
    result.bits.resize(n_bytes, 0);

    int threshold = valid_frames / 2; // strictly more than half
    for (int b = 0; b < n_bits; ++b) {
        if (votes[b] > threshold) {
            result.set_bit(b);
        }
    }

    return result;
}
