#pragma once

#include "hash.hpp"
#include "threadpool.hpp"

#include <functional>
#include <string>
#include <vector>
#include <limits>
#include <cmath>

// --- Input ---

struct ClusterInput {
    std::string algorithm;
    int hash_size;
    int hash_bits;
    std::vector<std::string> files;
    std::vector<HashResult> hashes;
};

// --- Distance matrix ---

struct DistanceMatrix {
    std::vector<std::vector<int>> data;  // data[i][j] = hamming(files[i], files[j])
    int n;

    int get(int i, int j) const { return data[i][j]; }

    // Global stats
    int min_distance() const;
    int max_distance() const;
    double avg_distance() const;
};

DistanceMatrix compute_distance_matrix(
    const std::vector<HashResult>& hashes,
    int num_threads,
    std::function<void(int,int)> progress_cb = nullptr);

// --- Groups ---

struct GroupInfo {
    int id;
    std::vector<int> members;
    int max_internal_distance;
    double avg_internal_distance;
};

// Turns member lists into GroupInfo, sorting each list and filling in its
// internal distance stats. Those stats are O(|group|^2), so a single dominant
// group costs as much as the whole distance matrix — the groups are built in
// parallel on the caller's pool. IDs are the member lists' own indices.
std::vector<GroupInfo> build_groups(
    const std::vector<std::vector<int>>& members,
    const DistanceMatrix& dm,
    ThreadPool& pool);

// --- Dendrogram (hierarchical) ---

struct DendrogramStep {
    int step;
    int merged_a;  // group/node index
    int merged_b;
    int distance;
};

// --- HDBSCAN membership ---

struct MembershipInfo {
    int file;
    int group;       // -1 = noise
    double confidence;
};

struct CondensedTreeNode {
    int parent;
    int child;
    double lambda;   // 1/distance
    int child_size;
};

// --- Clustering algorithms ---

std::vector<GroupInfo> threshold_cluster(
    const DistanceMatrix& dm, int threshold, int num_threads = 1);

struct HierarchicalResult {
    std::vector<GroupInfo> groups;
    std::vector<DendrogramStep> dendrogram;
};

HierarchicalResult hierarchical_cluster(
    const DistanceMatrix& dm, int cut_height,
    const std::string& linkage = "complete",
    int num_threads = 1);

struct HdbscanResult {
    std::vector<GroupInfo> groups;
    std::vector<MembershipInfo> membership;
    std::vector<CondensedTreeNode> condensed_tree;
};

HdbscanResult hdbscan_cluster(
    const DistanceMatrix& dm, int min_cluster_size, int num_threads = 1);
