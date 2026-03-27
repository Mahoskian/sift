#include "hash.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

// wHash: wavelet hash via Haar wavelet transform.
//
// 1. Resize image to size x size grayscale (size must be power of 2).
// 2. Apply 2D Haar wavelet transform.
// 3. Keep top-left size x size coefficients (approximation + detail).
// 4. Compute median of coefficients (excluding DC).
// 5. Each bit = 1 if coefficient > median, else 0.
// Produces size*size bits.

// 1D Haar wavelet transform (one level), in-place.
// Takes array of length N (must be even), produces:
//   [avg0, avg1, ..., detail0, detail1, ...]
static void haar_1d(double* data, int N) {
    std::vector<double> tmp(N);
    int half = N / 2;
    double norm = 1.0 / std::sqrt(2.0);
    for (int i = 0; i < half; i++) {
        tmp[i]        = (data[2 * i] + data[2 * i + 1]) * norm;
        tmp[half + i] = (data[2 * i] - data[2 * i + 1]) * norm;
    }
    for (int i = 0; i < N; i++) data[i] = tmp[i];
}

// Full multi-level 1D Haar transform: repeatedly apply to the
// approximation portion until length 1.
static void haar_full_1d(double* data, int N) {
    for (int len = N; len >= 2; len /= 2) {
        haar_1d(data, len);
    }
}

// 2D Haar wavelet: apply full 1D transform to each row, then each column.
static void haar_2d(std::vector<double>& matrix, int N) {
    // Rows
    for (int r = 0; r < N; r++) {
        haar_full_1d(&matrix[r * N], N);
    }
    // Columns
    std::vector<double> col(N);
    for (int c = 0; c < N; c++) {
        for (int r = 0; r < N; r++) col[r] = matrix[r * N + c];
        haar_full_1d(col.data(), N);
        for (int r = 0; r < N; r++) matrix[r * N + c] = col[r];
    }
}

// Round up to next power of 2
static int next_pow2(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

HashResult whash(const std::string& path, int size) {
    // The image is resized to a power-of-2 size for the wavelet transform.
    // We use the hash size directly if it's a power of 2, otherwise round up.
    int img_size = next_pow2(size);
    // Use at least 2x the hash size for better frequency resolution
    if (img_size < size * 2) img_size = next_pow2(size * 2);
    if (img_size < 8) img_size = 8;

    auto gray = load_grayscale_resized(path, img_size, img_size);
    if (gray.empty()) return {{}, 0};

    // Convert to doubles
    std::vector<double> wavelet(img_size * img_size);
    for (int i = 0; i < img_size * img_size; i++) {
        wavelet[i] = static_cast<double>(gray[i]);
    }

    // 2D Haar wavelet transform
    haar_2d(wavelet, img_size);

    // Extract top-left size x size coefficients
    std::vector<double> low_freq;
    low_freq.reserve(size * size);
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            low_freq.push_back(wavelet[r * img_size + c]);
        }
    }

    // Median (excluding DC at index 0)
    std::vector<double> for_median(low_freq.begin() + 1, low_freq.end());
    std::sort(for_median.begin(), for_median.end());
    double median;
    int n = (int)for_median.size();
    if (n % 2 == 0) {
        median = (for_median[n / 2 - 1] + for_median[n / 2]) / 2.0;
    } else {
        median = for_median[n / 2];
    }

    // Build hash
    int total_bits = size * size;
    int total_bytes = (total_bits + 7) / 8;

    HashResult result;
    result.size = size;
    result.bits.resize(total_bytes, 0);

    for (int i = 0; i < total_bits; i++) {
        if (low_freq[i] > median) {
            result.set_bit(i);
        }
    }

    return result;
}
