#include "cluster.hpp"

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

// Hierarchical agglomerative clustering.
//
// 1. Start with N singleton clusters.
// 2. Find the two closest clusters (by linkage criterion).
// 3. Merge them, record the merge in the dendrogram.
// 4. Repeat until one cluster remains.
// 5. Cut the dendrogram at cut_height to produce groups.

// Linkage functions: given two clusters and the distance matrix,
// compute the inter-cluster distance.

static int single_linkage(const std::vector<int>& a, const std::vector<int>& b,
                          const DistanceMatrix& dm) {
    int mn = std::numeric_limits<int>::max();
    for (int i : a)
        for (int j : b)
            mn = std::min(mn, dm.get(i, j));
    return mn;
}

static int complete_linkage(const std::vector<int>& a, const std::vector<int>& b,
                            const DistanceMatrix& dm) {
    int mx = 0;
    for (int i : a)
        for (int j : b)
            mx = std::max(mx, dm.get(i, j));
    return mx;
}

static int average_linkage(const std::vector<int>& a, const std::vector<int>& b,
                           const DistanceMatrix& dm) {
    long long sum = 0;
    long long count = 0;
    for (int i : a)
        for (int j : b) {
            sum += dm.get(i, j);
            count++;
        }
    return (int)(sum / count);
}

using LinkageFn = int(*)(const std::vector<int>&, const std::vector<int>&, const DistanceMatrix&);

static GroupInfo make_group_from(int id, const std::vector<int>& members, const DistanceMatrix& dm) {
    GroupInfo g;
    g.id = id;
    g.members = members;
    g.max_internal_distance = 0;
    g.avg_internal_distance = 0.0;

    if (members.size() > 1) {
        long long sum = 0;
        long long count = 0;
        for (size_t i = 0; i < members.size(); i++) {
            for (size_t j = i + 1; j < members.size(); j++) {
                int d = dm.get(members[i], members[j]);
                g.max_internal_distance = std::max(g.max_internal_distance, d);
                sum += d;
                count++;
            }
        }
        g.avg_internal_distance = (double)sum / count;
    }

    return g;
}

HierarchicalResult hierarchical_cluster(
    const DistanceMatrix& dm, int cut_height, const std::string& linkage)
{
    int n = dm.n;

    LinkageFn link_fn = complete_linkage;
    if (linkage == "single") link_fn = single_linkage;
    else if (linkage == "average") link_fn = average_linkage;

    // Active clusters: each is a vector of file indices.
    // We use a vector of optional clusters — merged clusters become empty.
    std::vector<std::vector<int>> clusters(n);
    for (int i = 0; i < n; i++) clusters[i] = {i};

    std::vector<bool> active(n, true);
    std::vector<DendrogramStep> dendrogram;

    // N-1 merges to build the full tree
    for (int step = 0; step < n - 1; step++) {
        // Find closest pair of active clusters
        int best_i = -1, best_j = -1;
        int best_dist = std::numeric_limits<int>::max();

        for (int i = 0; i < (int)clusters.size(); i++) {
            if (!active[i]) continue;
            for (int j = i + 1; j < (int)clusters.size(); j++) {
                if (!active[j]) continue;
                int d = link_fn(clusters[i], clusters[j], dm);
                if (d < best_dist) {
                    best_dist = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Record merge
        dendrogram.push_back({step, best_i, best_j, best_dist});

        // Merge j into i
        clusters[best_i].insert(
            clusters[best_i].end(),
            clusters[best_j].begin(),
            clusters[best_j].end());
        active[best_j] = false;

        // Create new node for dendrogram tracking
        // (the merged cluster keeps index best_i)
    }

    // Cut: replay merges, stop merging when distance > cut_height
    std::vector<std::vector<int>> cut_clusters(n);
    for (int i = 0; i < n; i++) cut_clusters[i] = {i};
    std::vector<bool> cut_active(n, true);

    for (const auto& merge : dendrogram) {
        if (merge.distance > cut_height) break;

        int a = merge.merged_a;
        int b = merge.merged_b;

        // Find current roots: after previous merges, a or b may have been
        // merged into something else. We track by replaying the same indices.
        if (!cut_active[a] || !cut_active[b]) continue;

        cut_clusters[a].insert(
            cut_clusters[a].end(),
            cut_clusters[b].begin(),
            cut_clusters[b].end());
        cut_active[b] = false;
    }

    // Collect resulting groups
    HierarchicalResult result;
    result.dendrogram = dendrogram;

    int gid = 0;
    for (int i = 0; i < (int)cut_clusters.size(); i++) {
        if (!cut_active[i]) continue;
        auto& members = cut_clusters[i];
        std::sort(members.begin(), members.end());
        result.groups.push_back(make_group_from(gid++, members, dm));
    }

    return result;
}
