#pragma once

#include <string>
#include <vector>

struct ProjectionResult {
    int dims;
    std::string method;
    std::vector<std::string> files;
    // points[i] = coordinates of file i, length = dims
    std::vector<std::vector<double>> points;
    // PCA-specific: variance explained by each component (fraction)
    std::vector<double> variance_explained;
};

// Convert hash hex strings to binary feature vectors (each bit → 0.0/1.0).
// Returns matrix[n_samples][n_features].
std::vector<std::vector<double>> hashes_to_features(
    const std::vector<std::string>& hex_hashes, int hash_bits);

// PCA: project into top-k dimensions.
// Uses dual PCA (Gram matrix) — efficient when n_samples << n_features.
ProjectionResult pca_project(
    const std::vector<std::vector<double>>& features,
    int dims);

// t-SNE: project into dims dimensions.
// perplexity: effective number of neighbors (typically 5-50).
// max_iter: gradient descent iterations.
// learning_rate: step size.
ProjectionResult tsne_project(
    const std::vector<std::vector<double>>& features,
    int dims,
    double perplexity = 30.0,
    int max_iter = 1000,
    double learning_rate = 200.0);
