#pragma once

#include "cluster.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace io {

// Scan directory for supported image files.
std::vector<std::filesystem::path> scan_images(const std::filesystem::path& dir);

// Parse hash JSON (output of `sift hash`) into ClusterInput.
ClusterInput parse_hash_json(const std::string& json_str);

// Read entire file to string.
std::string read_file(const std::string& path);

} // namespace io
