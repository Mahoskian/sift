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

// ── Parsed output formats (for sift remove) ─────────────────────────────────

struct ParsedProjection {
    std::string algorithm;
    int hash_size = 0;
    int hash_bits = 0;
    std::string method;
    int dims = 0;
    std::vector<std::string> files;
    std::vector<std::vector<double>> points;
    std::vector<double> variance_explained;
};

struct ParsedCluster {
    std::string algorithm;
    int hash_size = 0;
    int hash_bits = 0;
    std::string method;
    std::string raw_params;              // verbatim params JSON object
    std::vector<std::string> files;
    std::vector<std::vector<int>> distance_matrix;
    std::vector<GroupInfo> groups;       // already-filtered groups (as written)
    std::vector<int> ungrouped;
    bool has_membership = false;
    std::vector<MembershipInfo> membership;
};

ParsedProjection parse_project_json(const std::string& json_str);
ParsedCluster    parse_cluster_json(const std::string& json_str);

} // namespace io
