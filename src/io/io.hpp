#pragma once

#include "cluster.hpp"

#include <filesystem>
#include <vector>

// Returns paths of all supported image files in the given directory.
namespace io {
    std::vector<std::filesystem::path> scan_images(const std::filesystem::path& dir);
} // namespace sift

// Moves grouped files into subdirectories of outputDir.
void writeGroups(const std::vector<Group>& groups, const std::filesystem::path& outputDir);
