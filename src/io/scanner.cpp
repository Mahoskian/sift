#include "io.hpp"
#include <unordered_set>
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const std::unordered_set<std::string> IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tga", ".gif"
};

static const std::unordered_set<std::string> VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".flv", ".wmv"
};

static std::vector<fs::path> scan_dir(
    const fs::path& dir,
    const std::unordered_set<std::string>& exts)
{
    std::vector<fs::path> paths;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (exts.count(ext)) paths.push_back(entry.path());
    }
    return paths;
}

namespace io {

std::vector<fs::path> scan_images(const fs::path& dir) {
    return scan_dir(dir, IMAGE_EXTS);
}

std::vector<fs::path> scan_videos(const fs::path& dir) {
    return scan_dir(dir, VIDEO_EXTS);
}

std::vector<fs::path> scan_media(const fs::path& dir) {
    std::unordered_set<std::string> all_exts;
    all_exts.insert(IMAGE_EXTS.begin(), IMAGE_EXTS.end());
    all_exts.insert(VIDEO_EXTS.begin(), VIDEO_EXTS.end());
    return scan_dir(dir, all_exts);
}

} // namespace io
