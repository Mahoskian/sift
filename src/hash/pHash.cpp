#include "hash.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>

// pHash: perceptual hash via Discrete Cosine Transform.
//
// 1. Resize image to size x size grayscale.
// 2. Compute 2D DCT of the image.
// 3. Keep top-left size x size DCT coefficients (low frequencies).
// 4. Compute median of those coefficients (excluding DC term).
// 5. Each bit = 1 if coefficient > median, else 0.
// Produces size*size bits.

// Type-II DCT on a 1D signal of length N, in-place.
static void dct_1d(double* data, int N) {
    std::vector<double> out(N);
    double scale = M_PI / N;
    for (int k = 0; k < N; k++) {
        double sum = 0.0;
        for (int n = 0; n < N; n++) {
            sum += data[n] * std::cos(scale * (n + 0.5) * k);
        }
        out[k] = sum;
    }
    for (int k = 0; k < N; k++) data[k] = out[k];
}

// 2D DCT: apply 1D DCT to each row, then each column.
static void dct_2d(std::vector<double>& matrix, int N) {
    // Rows
    for (int r = 0; r < N; r++) {
        dct_1d(&matrix[r * N], N);
    }
    // Columns
    std::vector<double> col(N);
    for (int c = 0; c < N; c++) {
        for (int r = 0; r < N; r++) col[r] = matrix[r * N + c];
        dct_1d(col.data(), N);
        for (int r = 0; r < N; r++) matrix[r * N + c] = col[r];
    }
}

HashResult phash(const std::string& path, int size) {
    // For pHash, we resize to a larger image (4x the hash size) then DCT
    // and keep the top-left size x size coefficients.
    // This is the standard approach — the larger input gives the DCT
    // more frequency resolution.
    int img_size = size * 4;
    if (img_size < 32) img_size = 32;  // minimum 32x32 input

    auto gray = load_grayscale_resized(path, img_size, img_size);
    if (gray.empty()) return {{}, 0};

    // Convert to doubles
    std::vector<double> dct_matrix(img_size * img_size);
    for (int i = 0; i < img_size * img_size; i++) {
        dct_matrix[i] = static_cast<double>(gray[i]);
    }

    // 2D DCT
    dct_2d(dct_matrix, img_size);

    // Extract top-left size x size coefficients (excluding DC at [0,0])
    std::vector<double> low_freq;
    low_freq.reserve(size * size);
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            low_freq.push_back(dct_matrix[r * img_size + c]);
        }
    }

    // Median (excluding DC term at index 0)
    std::vector<double> for_median(low_freq.begin() + 1, low_freq.end());
    std::sort(for_median.begin(), for_median.end());
    double median;
    int n = (int)for_median.size();
    if (n % 2 == 0) {
        median = (for_median[n / 2 - 1] + for_median[n / 2]) / 2.0;
    } else {
        median = for_median[n / 2];
    }

    // Build hash: bit=1 if coefficient > median
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
