#include "cluster.hpp"
#include "parallel.hpp"
#include "threadpool.hpp"

#include <algorithm>
#include <atomic>

int DistanceMatrix::min_distance() const {
    int mn = std::numeric_limits<int>::max();
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            mn = std::min(mn, data[i][j]);
    return n > 1 ? mn : 0;
}

int DistanceMatrix::max_distance() const {
    int mx = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            mx = std::max(mx, data[i][j]);
    return mx;
}

double DistanceMatrix::avg_distance() const {
    if (n < 2) return 0.0;
    long long sum = 0;
    long long count = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            sum += data[i][j];
            count++;
        }
    return (double)sum / count;
}

DistanceMatrix compute_distance_matrix(
    const std::vector<HashResult>& hashes,
    int num_threads,
    std::function<void(int,int)> progress_cb)
{
    int n = (int)hashes.size();
    DistanceMatrix dm;
    dm.n = n;
    dm.data.resize(n, std::vector<int>(n, 0));

    ThreadPool pool(num_threads);
    std::atomic<int> rows_done{0};

    // Row i owns cells (i,j) and (j,i) for every j > i, so no cell is written
    // twice even though rows share the mirrored halves of the matrix.
    parallel_for(pool, n, [&](int i) {
        for (int j = i + 1; j < n; j++) {
            int d = HashResult::hamming(hashes[i], hashes[j]);
            dm.data[i][j] = d;
            dm.data[j][i] = d;
        }
        int done = ++rows_done;
        if (progress_cb) progress_cb(done, n);
    });

    return dm;
}

std::vector<GroupInfo> build_groups(
    const std::vector<std::vector<int>>& members,
    const DistanceMatrix& dm,
    ThreadPool& pool)
{
    std::vector<GroupInfo> groups(members.size());

    parallel_for(pool, (int)members.size(), [&](int g) {
        GroupInfo& info = groups[g];
        info.id = g;
        info.members = members[g];
        info.max_internal_distance = 0;
        info.avg_internal_distance = 0.0;
        std::sort(info.members.begin(), info.members.end());

        if (info.members.size() < 2) return;

        long long sum = 0;
        long long count = 0;
        for (size_t i = 0; i < info.members.size(); i++) {
            for (size_t j = i + 1; j < info.members.size(); j++) {
                int d = dm.get(info.members[i], info.members[j]);
                info.max_internal_distance = std::max(info.max_internal_distance, d);
                sum += d;
                count++;
            }
        }
        info.avg_internal_distance = (double)sum / count;
    });

    return groups;
}
