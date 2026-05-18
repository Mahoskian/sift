#pragma once

#include "cluster.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace io {

// Scan directory recursively for supported image files.
std::vector<std::filesystem::path> scan_images(const std::filesystem::path& dir);

// Scan directory recursively for supported video files.
std::vector<std::filesystem::path> scan_videos(const std::filesystem::path& dir);

// Scan directory recursively for both images and videos.
std::vector<std::filesystem::path> scan_media(const std::filesystem::path& dir);

// Parse hash JSON (output of `sift hash`) into ClusterInput.
ClusterInput parse_hash_json(const std::string& json_str);

// Read entire file to string.
std::string read_file(const std::string& path);

} // namespace io
