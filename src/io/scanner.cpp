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

namespace io {

    std::vector<fs::path> scan_images(const fs::path& dir) {

        std::vector<fs::path> paths;

        for (const auto& entry : fs::recursive_directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;

            auto ext = entry.path().extension().string();

            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (IMAGE_EXTS.count(ext)) {
                paths.push_back(entry.path());
            }
        }

        return paths;
    }

} // namespace io
