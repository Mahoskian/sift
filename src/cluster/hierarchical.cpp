#include "cluster.hpp"
#include "parallel.hpp"

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

HierarchicalResult hierarchical_cluster(
    const DistanceMatrix& dm, int cut_height, const std::string& linkage,
    int num_threads)
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

    ThreadPool pool(num_threads);

    // Per-row winner of the closest-pair scan, reused across every merge step.
    std::vector<int> row_j(n);
    std::vector<int> row_dist(n);

    // N-1 merges to build the full tree
    for (int step = 0; step < n - 1; step++) {
        // Find closest pair of active clusters. Each row i scans only j > i and
        // records its own best, so rows never contend; the winners are then
        // reduced in ascending i with the same strict < the serial scan used,
        // which reproduces its exact tie-breaking (lowest i, then lowest j).
        parallel_for(pool, n, [&](int i) {
            row_dist[i] = std::numeric_limits<int>::max();
            row_j[i] = -1;
            if (!active[i]) return;
            for (int j = i + 1; j < n; j++) {
                if (!active[j]) continue;
                int d = link_fn(clusters[i], clusters[j], dm);
                if (d < row_dist[i]) {
                    row_dist[i] = d;
                    row_j[i] = j;
                }
            }
        }, 8);

        int best_i = -1, best_j = -1;
        int best_dist = std::numeric_limits<int>::max();
        for (int i = 0; i < n; i++) {
            if (row_j[i] >= 0 && row_dist[i] < best_dist) {
                best_dist = row_dist[i];
                best_i = i;
                best_j = row_j[i];
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

    std::vector<std::vector<int>> surviving;
    for (int i = 0; i < (int)cut_clusters.size(); i++)
        if (cut_active[i]) surviving.push_back(std::move(cut_clusters[i]));

    result.groups = build_groups(surviving, dm, pool);
    return result;
}
