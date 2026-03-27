#include "cluster.hpp"

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

// HDBSCAN: Hierarchical Density-Based Spatial Clustering.
//
// 1. Compute core distances (distance to min_cluster_size-th nearest neighbor).
// 2. Build mutual reachability graph (edge weight = max(core_a, core_b, dist(a,b))).
// 3. Build minimum spanning tree of the mutual reachability graph.
// 4. Build cluster hierarchy from the MST (like single-linkage).
// 5. Condense the tree: collapse clusters smaller than min_cluster_size.
// 6. Extract stable clusters using excess of mass.
// 7. Assign confidence scores.

// --- Minimum spanning tree (Prim's) ---

struct Edge {
    int u, v;
    int weight;
};

static std::vector<Edge> mst_prim(const std::vector<std::vector<int>>& graph, int n) {
    std::vector<bool> in_tree(n, false);
    std::vector<int> min_edge(n, std::numeric_limits<int>::max());
    std::vector<int> min_from(n, -1);
    min_edge[0] = 0;

    std::vector<Edge> mst;
    mst.reserve(n - 1);

    for (int iter = 0; iter < n; iter++) {
        // Find cheapest vertex not yet in tree
        int u = -1;
        for (int i = 0; i < n; i++)
            if (!in_tree[i] && (u == -1 || min_edge[i] < min_edge[u]))
                u = i;

        in_tree[u] = true;
        if (min_from[u] != -1)
            mst.push_back({min_from[u], u, min_edge[u]});

        // Update neighbors
        for (int v = 0; v < n; v++) {
            if (!in_tree[v] && graph[u][v] < min_edge[v]) {
                min_edge[v] = graph[u][v];
                min_from[v] = u;
            }
        }
    }

    return mst;
}

// --- Union-Find ---

struct UF {
    std::vector<int> parent, sz;
    UF(int n) : parent(n), sz(n, 1) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    int unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return a;
        if (sz[a] < sz[b]) std::swap(a, b);
        parent[b] = a; sz[a] += sz[b];
        return a;
    }
    int size(int x) { return sz[find(x)]; }
};

// --- Condensed tree node (internal representation) ---

struct CNode {
    int parent;
    int child;
    double lambda;
    int child_size;
};

static GroupInfo make_hdbscan_group(int id, const std::vector<int>& members, const DistanceMatrix& dm) {
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

HdbscanResult hdbscan_cluster(const DistanceMatrix& dm, int min_cluster_size) {
    int n = dm.n;

    // 1. Core distances: for each point, distance to its (min_cluster_size-1)-th
    //    nearest neighbor (k-th nearest, 0-indexed, excluding self)
    int k = std::min(min_cluster_size - 1, n - 1);
    if (k < 1) k = 1;

    std::vector<int> core_dist(n);
    for (int i = 0; i < n; i++) {
        std::vector<int> dists;
        dists.reserve(n - 1);
        for (int j = 0; j < n; j++)
            if (i != j) dists.push_back(dm.get(i, j));
        std::sort(dists.begin(), dists.end());
        core_dist[i] = dists[std::min(k - 1, (int)dists.size() - 1)];
    }

    // 2. Mutual reachability graph
    //    mr_dist(a, b) = max(core_dist[a], core_dist[b], dist(a, b))
    std::vector<std::vector<int>> mr_graph(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            int d = std::max({core_dist[i], core_dist[j], dm.get(i, j)});
            mr_graph[i][j] = d;
            mr_graph[j][i] = d;
        }

    // 3. Minimum spanning tree
    auto mst = mst_prim(mr_graph, n);
    std::sort(mst.begin(), mst.end(),
              [](const Edge& a, const Edge& b) { return a.weight < b.weight; });

    // 4. Build single-linkage hierarchy from MST
    //    Process edges in order of weight. Each merge creates a new internal node.
    //    Internal nodes are indexed n, n+1, n+2, ...
    int next_node = n;
    UF uf(2 * n);  // enough room for internal nodes
    // Track which hierarchy node each UF component maps to
    std::vector<int> comp_node(2 * n);
    std::iota(comp_node.begin(), comp_node.end(), 0);

    struct HierEdge {
        int parent;       // new internal node
        int child_a;      // hierarchy node
        int child_b;      // hierarchy node
        int child_a_size;
        int child_b_size;
        double lambda;    // 1/distance
    };
    std::vector<HierEdge> hierarchy;

    for (auto& e : mst) {
        int ra = uf.find(e.u);
        int rb = uf.find(e.v);
        if (ra == rb) continue;

        int node_a = comp_node[ra];
        int node_b = comp_node[rb];
        int size_a = uf.size(ra);
        int size_b = uf.size(rb);

        int new_node = next_node++;
        double lambda = (e.weight > 0) ? 1.0 / e.weight : std::numeric_limits<double>::max();

        hierarchy.push_back({new_node, node_a, node_b, size_a, size_b, lambda});

        int root = uf.unite(ra, rb);
        comp_node[root] = new_node;
    }

    // 5. Condense the tree: walk hierarchy top-down.
    //    When a split produces a child smaller than min_cluster_size,
    //    that child's points "fall out" of the parent cluster instead of
    //    forming their own cluster.

    // For each hierarchy node, track its members (leaf points).
    // Internal nodes that survive condensation are "real clusters".
    // We'll rebuild by walking top-down.

    // First, build a map from internal node → its hierarchy entry
    std::vector<int> left_child(next_node, -1);
    std::vector<int> right_child(next_node, -1);
    std::vector<int> left_size(next_node, 0);
    std::vector<int> right_size(next_node, 0);
    std::vector<double> split_lambda(next_node, 0.0);

    for (auto& h : hierarchy) {
        left_child[h.parent] = h.child_a;
        right_child[h.parent] = h.child_b;
        left_size[h.parent] = h.child_a_size;
        right_size[h.parent] = h.child_b_size;
        split_lambda[h.parent] = h.lambda;
    }

    // Collect leaf members for each node (bottom-up)
    std::vector<std::vector<int>> node_members(next_node);
    for (int i = 0; i < n; i++) node_members[i] = {i};
    for (auto& h : hierarchy) {
        auto& pm = node_members[h.parent];
        pm.insert(pm.end(), node_members[h.child_a].begin(), node_members[h.child_a].end());
        pm.insert(pm.end(), node_members[h.child_b].begin(), node_members[h.child_b].end());
    }

    // Build condensed tree
    std::vector<CondensedTreeNode> condensed_tree;

    // Walk top-down: for each internal node, if both children >= min_cluster_size,
    // it's a real split. Otherwise, the small child's points fall out as noise
    // candidates, and the parent continues as one cluster.

    int root_node = next_node - 1;  // top of hierarchy

    // BFS / recursive condensation
    // We'll use a stack-based approach.
    // condensed_cluster_id[node] = which condensed cluster this node belongs to
    std::vector<int> condensed_id(next_node, -1);
    int next_condensed = 0;

    struct StackEntry {
        int node;
        int parent_cluster;  // condensed cluster id of parent, or -1
    };

    std::vector<StackEntry> stack;
    stack.push_back({root_node, -1});

    // Each condensed cluster: track its birth lambda and members with their fall-out lambda
    struct CondensedCluster {
        int id;
        double birth_lambda;
        std::vector<std::pair<int, double>> fallen_points;  // (file_idx, lambda)
        std::vector<int> child_clusters;
        double stability;
    };

    std::vector<CondensedCluster> condensed_clusters;

    // Assign root a cluster
    int root_cluster = next_condensed++;
    condensed_clusters.push_back({root_cluster, 0.0, {}, {}, 0.0});

    stack.clear();
    stack.push_back({root_node, root_cluster});

    while (!stack.empty()) {
        auto [node, parent_cid] = stack.back();
        stack.pop_back();

        if (node < n) {
            // Leaf — this point belongs to parent_cid
            continue;
        }

        int lc = left_child[node];
        int rc = right_child[node];
        int ls = left_size[node];
        int rs = right_size[node];
        double lam = split_lambda[node];

        bool left_survives = (ls >= min_cluster_size);
        bool right_survives = (rs >= min_cluster_size);

        if (left_survives && right_survives) {
            // Real split: create two new condensed clusters
            int lcid = next_condensed++;
            int rcid = next_condensed++;
            condensed_clusters.push_back({lcid, lam, {}, {}, 0.0});
            condensed_clusters.push_back({rcid, lam, {}, {}, 0.0});
            condensed_clusters[parent_cid].child_clusters.push_back(lcid);
            condensed_clusters[parent_cid].child_clusters.push_back(rcid);

            condensed_tree.push_back({parent_cid, lcid, lam, ls});
            condensed_tree.push_back({parent_cid, rcid, lam, rs});

            stack.push_back({lc, lcid});
            stack.push_back({rc, rcid});
        } else if (!left_survives && !right_survives) {
            // Both too small — all points fall out of parent cluster at this lambda
            for (int pt : node_members[lc])
                condensed_clusters[parent_cid].fallen_points.push_back({pt, lam});
            for (int pt : node_members[rc])
                condensed_clusters[parent_cid].fallen_points.push_back({pt, lam});
        } else {
            // One survives, one doesn't — small child's points fall out
            int surviving = left_survives ? lc : rc;
            int falling = left_survives ? rc : lc;

            for (int pt : node_members[falling])
                condensed_clusters[parent_cid].fallen_points.push_back({pt, lam});

            // Surviving child continues as parent_cid
            stack.push_back({surviving, parent_cid});
        }
    }

    // 6. Compute stability for each condensed cluster.
    //    stability = sum over all points that fell out: (lambda_fall - lambda_birth)
    for (auto& cc : condensed_clusters) {
        double stab = 0.0;
        for (auto& [pt, lam] : cc.fallen_points) {
            stab += (lam - cc.birth_lambda);
        }
        cc.stability = stab;
    }

    // 7. Extract clusters: bottom-up, each cluster is selected if its stability
    //    exceeds the sum of its children's stabilities.
    //    Process in reverse order (leaves first).
    std::vector<bool> selected(next_condensed, false);
    std::vector<double> subtree_stability(next_condensed, 0.0);

    for (int i = next_condensed - 1; i >= 0; i--) {
        auto& cc = condensed_clusters[i];
        if (cc.child_clusters.empty()) {
            // Leaf cluster: always selected initially
            selected[i] = true;
            subtree_stability[i] = cc.stability;
        } else {
            double children_stab = 0.0;
            for (int child : cc.child_clusters)
                children_stab += subtree_stability[child];

            if (cc.stability >= children_stab) {
                // This cluster is better than its children
                selected[i] = true;
                subtree_stability[i] = cc.stability;
                // Deselect all descendants
                std::vector<int> desc_stack = cc.child_clusters;
                while (!desc_stack.empty()) {
                    int d = desc_stack.back();
                    desc_stack.pop_back();
                    selected[d] = false;
                    for (int dd : condensed_clusters[d].child_clusters)
                        desc_stack.push_back(dd);
                }
            } else {
                subtree_stability[i] = children_stab;
            }
        }
    }

    // Don't select the root — it's the "everything" cluster
    selected[root_cluster] = false;

    // 8. Assign points to clusters and compute confidence.
    //    Walk the condensed tree: each point belongs to the deepest selected cluster
    //    that contains it.

    // First: for each selected cluster, collect all its points
    // A point belongs to a condensed cluster if it fell out of it, OR if it's in
    // a descendant that wasn't selected (and bubbles up).

    // Simple approach: for each point, find which selected cluster it ends up in.
    std::vector<int> point_cluster(n, -1);
    std::vector<double> point_lambda(n, 0.0);

    // Traverse: for each selected cluster, the points that fell out of it belong to it.
    // Points that fell out of non-selected clusters bubble up.
    // Process top-down: assign to the deepest selected ancestor.

    std::vector<int> topo_order;
    {
        std::vector<int> ts;
        ts.push_back(root_cluster);
        for (size_t idx = 0; idx < ts.size(); idx++) {
            int cid = ts[idx];
            for (int child : condensed_clusters[cid].child_clusters)
                ts.push_back(child);
        }
        topo_order = ts;
    }

    // For each condensed cluster, track the selected ancestor
    std::vector<int> selected_ancestor(next_condensed, -1);
    for (int cid : topo_order) {
        if (selected[cid]) {
            selected_ancestor[cid] = cid;
        }
        // Propagate to children
        for (int child : condensed_clusters[cid].child_clusters) {
            if (selected[child]) {
                selected_ancestor[child] = child;
            } else {
                selected_ancestor[child] = selected_ancestor[cid];
            }
        }
    }

    // Assign fallen points
    for (int cid = 0; cid < next_condensed; cid++) {
        int assign_to = selected_ancestor[cid];
        for (auto& [pt, lam] : condensed_clusters[cid].fallen_points) {
            point_cluster[pt] = assign_to;
            point_lambda[pt] = lam;
        }
    }

    // Points still unassigned: they survived to a leaf cluster without falling out.
    // Assign them to the deepest selected cluster they belong to.
    // These are points in leaf clusters of the condensed tree.
    for (int cid = 0; cid < next_condensed; cid++) {
        if (!condensed_clusters[cid].child_clusters.empty()) continue;
        // Leaf condensed cluster — get all members from the original hierarchy
        // that weren't captured as fallen_points
        // Actually, in a leaf condensed cluster, any remaining points
        // are the ones that never fell out — they belong to this cluster
        // if it's selected, or to its selected ancestor.
        // Points remaining in leaf clusters are handled by the
        // cc_all_points sweep below.
    }

    // Rebuild: track all members of each condensed cluster (including inherited from children)
    std::vector<std::vector<int>> cc_all_points(next_condensed);
    // Bottom-up: leaf condensed clusters inherit from their hierarchy subtree
    for (int i = next_condensed - 1; i >= 0; i--) {
        for (auto& [pt, lam] : condensed_clusters[i].fallen_points)
            cc_all_points[i].push_back(pt);
        for (int child : condensed_clusters[i].child_clusters)
            for (int pt : cc_all_points[child])
                cc_all_points[i].push_back(pt);
    }

    // Assign any unassigned points
    for (int cid = 0; cid < next_condensed; cid++) {
        if (!selected[cid]) continue;
        for (int pt : cc_all_points[cid]) {
            if (point_cluster[pt] == -1 || point_cluster[pt] == cid) {
                point_cluster[pt] = cid;
            }
        }
    }

    // Map condensed cluster IDs to sequential group IDs
    std::vector<int> cluster_to_group(next_condensed, -1);
    int num_groups = 0;
    for (int i = 0; i < next_condensed; i++)
        if (selected[i])
            cluster_to_group[i] = num_groups++;

    // 9. Compute confidence scores.
    //    confidence = (lambda_point - lambda_birth) / (lambda_max - lambda_birth)
    //    for the cluster the point belongs to.
    std::vector<double> cluster_max_lambda(next_condensed, 0.0);
    for (int i = 0; i < n; i++) {
        int cid = point_cluster[i];
        if (cid >= 0)
            cluster_max_lambda[cid] = std::max(cluster_max_lambda[cid], point_lambda[i]);
    }

    // Build output
    HdbscanResult result;

    // Membership
    for (int i = 0; i < n; i++) {
        int cid = point_cluster[i];
        int gid = (cid >= 0) ? cluster_to_group[cid] : -1;
        double conf = 0.0;
        if (cid >= 0) {
            double birth = condensed_clusters[cid].birth_lambda;
            double max_lam = cluster_max_lambda[cid];
            if (max_lam > birth)
                conf = (point_lambda[i] - birth) / (max_lam - birth);
            else
                conf = 1.0;
        }
        result.membership.push_back({i, gid, std::min(1.0, std::max(0.0, conf))});
    }

    // Groups
    std::vector<std::vector<int>> group_members(num_groups);
    for (int i = 0; i < n; i++) {
        int gid = result.membership[i].group;
        if (gid >= 0)
            group_members[gid].push_back(i);
    }

    for (int g = 0; g < num_groups; g++) {
        std::sort(group_members[g].begin(), group_members[g].end());
        result.groups.push_back(make_hdbscan_group(g, group_members[g], dm));
    }

    // Condensed tree output
    result.condensed_tree.clear();
    for (auto& ct : condensed_tree) {
        result.condensed_tree.push_back(ct);
    }

    return result;
}
