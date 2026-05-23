#include "project.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>

// Jacobi eigenvalue algorithm for symmetric matrices.
// Finds ALL eigenvalues and eigenvectors of an n x n symmetric matrix.
// Returns eigenvalues (descending) and corresponding eigenvectors (columns).

struct EigenResult {
    std::vector<double> values;
    std::vector<std::vector<double>> vectors;  // vectors[i] = i-th eigenvector
};

static EigenResult jacobi_eigen(std::vector<std::vector<double>> A, int n, int max_iter = 200) {
    // Initialize eigenvectors as identity
    std::vector<std::vector<double>> V(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) V[i][i] = 1.0;

    for (int iter = 0; iter < max_iter; iter++) {
        // Find largest off-diagonal element
        int p = 0, q = 1;
        double max_val = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (std::abs(A[i][j]) > max_val) {
                    max_val = std::abs(A[i][j]);
                    p = i; q = j;
                }
            }
        }

        // Convergence check
        if (max_val < 1e-12) break;

        // Compute rotation angle
        double theta;
        if (std::abs(A[p][p] - A[q][q]) < 1e-15) {
            theta = M_PI / 4.0;
        } else {
            theta = 0.5 * std::atan2(2.0 * A[p][q], A[p][p] - A[q][q]);
        }

        double c = std::cos(theta);
        double s = std::sin(theta);

        // Apply Givens rotation to A
        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
        A[p][p] = c * c * app + 2 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app - 2 * s * c * apq + c * c * aqq;
        A[p][q] = A[q][p] = 0.0;

        for (int i = 0; i < n; i++) {
            if (i == p || i == q) continue;
            double aip = A[i][p], aiq = A[i][q];
            A[i][p] = A[p][i] = c * aip + s * aiq;
            A[i][q] = A[q][i] = -s * aip + c * aiq;
        }

        // Update eigenvectors
        for (int i = 0; i < n; i++) {
            double vip = V[i][p], viq = V[i][q];
            V[i][p] = c * vip + s * viq;
            V[i][q] = -s * vip + c * viq;
        }
    }

    // Extract eigenvalues and sort descending
    EigenResult result;
    std::vector<std::pair<double, int>> evals(n);
    for (int i = 0; i < n; i++) evals[i] = {A[i][i], i};
    std::sort(evals.begin(), evals.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    result.values.resize(n);
    result.vectors.resize(n, std::vector<double>(n));
    for (int k = 0; k < n; k++) {
        result.values[k] = evals[k].first;
        int idx = evals[k].second;
        for (int i = 0; i < n; i++)
            result.vectors[k][i] = V[i][idx];
    }

    return result;
}

std::vector<std::vector<double>> hashes_to_features(
    const std::vector<std::string>& hex_hashes, int hash_bits)
{
    int n = (int)hex_hashes.size();
    std::vector<std::vector<double>> features(n, std::vector<double>(hash_bits));

    for (int i = 0; i < n; i++) {
        const auto& hex = hex_hashes[i];
        int bit = 0;
        for (size_t j = 0; j < hex.size(); j++) {
            unsigned int nibble;
            std::sscanf(hex.c_str() + j, "%1x", &nibble);
            for (int b = 3; b >= 0 && bit < hash_bits; b--, bit++) {
                features[i][bit] = (nibble >> b) & 1 ? 1.0 : 0.0;
            }
        }
    }

    return features;
}

ProjectionResult pca_project(
    const std::vector<std::vector<double>>& features, int dims,
    std::function<void(int,int)> progress_cb)
{
    int n = (int)features.size();
    int d = n > 0 ? (int)features[0].size() : 0;
    int k = std::min(dims, std::min(n, d));

    // Center the data
    std::vector<double> mean(d, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            mean[j] += features[i][j];
    for (int j = 0; j < d; j++) mean[j] /= n;

    std::vector<std::vector<double>> centered(n, std::vector<double>(d));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            centered[i][j] = features[i][j] - mean[j];

    // Dual PCA: compute Gram matrix G = X * X^T (n x n)
    // Much smaller than covariance matrix when n << d
    std::vector<std::vector<double>> G(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double dot = 0.0;
            for (int f = 0; f < d; f++)
                dot += centered[i][f] * centered[j][f];
            G[i][j] = dot;
            G[j][i] = dot;
        }
        if (progress_cb) progress_cb(i + 1, n);
    }

    // Eigendecomposition of G
    auto eigen = jacobi_eigen(G, n);

    // PC scores: Z[i][k] = eigenvector_k[i] * sqrt(eigenvalue_k)
    // (the i-th sample's coordinate on the k-th principal component)
    ProjectionResult result;
    result.dims = k;
    result.method = "pca";
    result.points.resize(n, std::vector<double>(k));

    double total_variance = 0.0;
    for (int i = 0; i < n; i++)
        total_variance += std::max(0.0, eigen.values[i]);

    result.variance_explained.resize(k);
    for (int c = 0; c < k; c++) {
        double ev = std::max(0.0, eigen.values[c]);
        double scale = std::sqrt(ev);
        result.variance_explained[c] = (total_variance > 0) ? ev / total_variance : 0.0;
        for (int i = 0; i < n; i++) {
            result.points[i][c] = eigen.vectors[c][i] * scale;
        }
    }

    return result;
}
