#pragma once

#include "threadpool.hpp"

#include <algorithm>
#include <atomic>
#include <future>
#include <vector>

// Runs fn(i) for every i in [0, n) across the pool's workers, returning once
// every index has been visited exactly once.
//
// Indices are handed out in chunks through an atomic cursor rather than sliced
// up front: most loops here are triangular (row i does n-i units of work), so a
// static split would leave the last workers idle for most of the pass.
//
// fn is called concurrently, so it must only touch state it owns for that
// index — writing element i of a pre-sized container needs no locking, but
// shared accumulators do (use a per-index partial and reduce afterwards, which
// also keeps the result deterministic).
//
// grain is the minimum number of indices worth giving a thread; when n is
// smaller than that the loop runs inline, since the hand-off costs more than
// the work saved.
template <typename F>
void parallel_for(ThreadPool& pool, int n, F&& fn, int grain = 1) {
    if (n <= 0) return;

    const int nt = std::min(pool.size(), n / std::max(1, grain));
    if (nt <= 1) {
        for (int i = 0; i < n; i++) fn(i);
        return;
    }

    const int chunk = std::max(grain, n / (nt * 8));
    std::atomic<int> cursor{0};

    auto worker = [&] {
        for (;;) {
            int begin = cursor.fetch_add(chunk);
            if (begin >= n) return;
            int end = std::min(begin + chunk, n);
            for (int i = begin; i < end; i++) fn(i);
        }
    };

    std::vector<std::future<void>> futures;
    futures.reserve(nt);
    for (int t = 0; t < nt; t++) futures.push_back(pool.submit(worker));
    for (auto& f : futures) f.get();
}
