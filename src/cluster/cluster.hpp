#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct MediaFile {
    std::string path;
    uint64_t    hash;
};

using Group = std::vector<MediaFile>;

// Returns the number of differing bits between two hashes.
inline int hamming(uint64_t a, uint64_t b) {
    return __builtin_popcountll(a ^ b);
}

// Groups files where hamming distance <= threshold.
std::vector<Group> thresholdCluster(const std::vector<MediaFile>& files, int threshold);

// HDBSCAN-based clustering (future).
std::vector<Group> hdbscanCluster(const std::vector<MediaFile>& files);
