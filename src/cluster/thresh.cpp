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

static GroupInfo make_group(int id, const std::vector<int>& members, const DistanceMatrix& dm) {
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

std::vector<GroupInfo> threshold_cluster(const DistanceMatrix& dm, int threshold) {
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
    std::vector<GroupInfo> groups;
    int gid = 0;
    for (auto& members : components) {
        std::sort(members.begin(), members.end());
        groups.push_back(make_group(gid++, members, dm));
    }

    return groups;
}
