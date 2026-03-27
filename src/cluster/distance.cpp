#include "cluster.hpp"
#include "threadpool.hpp"

#include <algorithm>
#include <future>

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

DistanceMatrix compute_distance_matrix(const std::vector<HashResult>& hashes, int num_threads) {
    int n = (int)hashes.size();
    DistanceMatrix dm;
    dm.n = n;
    dm.data.resize(n, std::vector<int>(n, 0));

    // Parallelize by rows
    ThreadPool pool(num_threads);
    std::vector<std::future<void>> futures;

    for (int i = 0; i < n; i++) {
        futures.push_back(pool.submit([&, i]() {
            for (int j = i + 1; j < n; j++) {
                int d = HashResult::hamming(hashes[i], hashes[j]);
                dm.data[i][j] = d;
                dm.data[j][i] = d;
            }
        }));
    }

    for (auto& f : futures) f.get();
    return dm;
}
