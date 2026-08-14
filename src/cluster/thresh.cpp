#include "cluster.hpp"

#include <vector>
#include <numeric>
#include <algorithm>

// Union-Find for connected components
struct UnionFind {
    std::vector<int> parent, rank;

    UnionFind(int n) : parent(n), rank(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // path compression
            x = parent[x];
        }
        return x;
    }

    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) rank[a]++;
    }
};

std::vector<GroupInfo> threshold_cluster(const DistanceMatrix& dm, int threshold,
                                         int num_threads) {
    int n = dm.n;
    UnionFind uf(n);

    // Connect all pairs within threshold
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (dm.get(i, j) <= threshold)
                uf.unite(i, j);

    // Collect connected components
    std::vector<std::vector<int>> components;
    std::vector<int> comp_map(n, -1);

    for (int i = 0; i < n; i++) {
        int root = uf.find(i);
        if (comp_map[root] == -1) {
            comp_map[root] = (int)components.size();
            components.push_back({});
        }
        components[comp_map[root]].push_back(i);
    }

    // Build GroupInfo for each component
    ThreadPool pool(num_threads);
    return build_groups(components, dm, pool);
}
