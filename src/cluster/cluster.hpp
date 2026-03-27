#pragma once

#include "hash.hpp"

#include <string>
#include <vector>

struct MediaFile {
    std::string path;
    HashResult  hash;
};

using Group = std::vector<MediaFile>;

// Groups files where hamming distance <= threshold.
std::vector<Group> thresholdCluster(const std::vector<MediaFile>& files, int threshold);

// HDBSCAN-based clustering (future).
std::vector<Group> hdbscanCluster(const std::vector<MediaFile>& files);
