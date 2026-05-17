#include "project.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

// t-SNE (t-distributed Stochastic Neighbor Embedding)
//
// 1. Compute pairwise squared Euclidean distances in high-D.
// 2. Convert to conditional probabilities p(j|i) using Gaussian kernel,
//    with per-point bandwidth σ_i chosen to match target perplexity.
// 3. Symmetrize: p_ij = (p(j|i) + p(i|j)) / (2n).
// 4. Initialize low-D positions randomly (from PCA if available).
// 5. Gradient descent: minimize KL(P || Q) where Q uses Student-t kernel.
// 6. Tricks: early exaggeration, momentum.

// Squared Euclidean distance between two vectors.
static double sq_dist(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

// Binary search for sigma_i that gives the target perplexity.
// P(j|i) = exp(-||x_i - x_j||^2 / (2 * sigma_i^2)) / sum_k(...)
// Perplexity = 2^(entropy of P(*|i))
static void compute_pij_row(
    const std::vector<double>& dists_sq,  // squared distances from point i to all others
    int i, int n, double target_perplexity,
    std::vector<double>& p_row)           // output: p(j|i) for all j
{
    double beta_min = -1e300;
    double beta_max = 1e300;
    double beta = 1.0;  // beta = 1 / (2 * sigma^2)
    double target_entropy = std::log(target_perplexity);

    p_row.resize(n, 0.0);

    for (int iter = 0; iter < 50; iter++) {
        // Compute P(j|i) and entropy
        double sum_exp = 0.0;
        for (int j = 0; j < n; j++) {
            if (j == i) { p_row[j] = 0.0; continue; }
            p_row[j] = std::exp(-dists_sq[j] * beta);
            sum_exp += p_row[j];
        }

        if (sum_exp < 1e-300) sum_exp = 1e-300;

        double entropy = 0.0;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            p_row[j] /= sum_exp;
            if (p_row[j] > 1e-300)
                entropy -= p_row[j] * std::log(p_row[j]);
        }

        double perp_diff = entropy - target_entropy;

        if (std::abs(perp_diff) < 1e-5) break;

        if (perp_diff > 0) {
            // Perplexity too high → increase beta (decrease sigma)
            beta_min = beta;
            beta = (beta_max == 1e300) ? beta * 2.0 : (beta + beta_max) / 2.0;
        } else {
            // Perplexity too low → decrease beta (increase sigma)
            beta_max = beta;
            beta = (beta_min == -1e300) ? beta / 2.0 : (beta + beta_min) / 2.0;
        }
    }
}

ProjectionResult tsne_project(
    const std::vector<std::vector<double>>& features,
    int dims,
    double perplexity,
    int max_iter,
    double learning_rate)
{
    int n = (int)features.size();

    // Clamp perplexity to sensible range
    perplexity = std::min(perplexity, (double)(n - 1));
    if (perplexity < 1.0) perplexity = 1.0;

    // 1. Pairwise squared distances in high-D
    std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            double d = sq_dist(features[i], features[j]);
            D[i][j] = d;
            D[j][i] = d;
        }

    // 2. Compute conditional probabilities p(j|i)
    std::vector<std::vector<double>> P(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        compute_pij_row(D[i], i, n, perplexity, P[i]);
    }

    // 3. Symmetrize: P_ij = (p(j|i) + p(i|j)) / (2n)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double sym = (P[i][j] + P[j][i]) / (2.0 * n);
            P[i][j] = sym;
            P[j][i] = sym;
        }
    }

    // 4. Initialize low-D positions via PCA
    auto pca_init = pca_project(features, dims);
    std::vector<std::vector<double>> Y = pca_init.points;

    // Scale down PCA initialization
    double max_coord = 0.0;
    for (auto& pt : Y)
        for (double v : pt)
            max_coord = std::max(max_coord, std::abs(v));
    if (max_coord > 0) {
        double scale = 1e-4 / max_coord;
        for (auto& pt : Y)
            for (double& v : pt)
                v *= scale;
    }

    // Gradient descent storage
    std::vector<std::vector<double>> gains(n, std::vector<double>(dims, 1.0));
    std::vector<std::vector<double>> Y_prev(n, std::vector<double>(dims, 0.0));
    std::vector<std::vector<double>> grad(n, std::vector<double>(dims, 0.0));

    // 5. Gradient descent
    for (int iter = 0; iter < max_iter; iter++) {
        // Early exaggeration: multiply P by 4 for first 250 iterations
        double exaggeration = (iter < 250) ? 4.0 : 1.0;
        double momentum = (iter < 250) ? 0.5 : 0.8;

        // Compute Q (Student-t with 1 DOF)
        std::vector<std::vector<double>> Q_num(n, std::vector<double>(n, 0.0));
        double Q_sum = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double d = sq_dist(Y[i], Y[j]);
                double q = 1.0 / (1.0 + d);
                Q_num[i][j] = q;
                Q_num[j][i] = q;
                Q_sum += 2.0 * q;
            }
        }
        if (Q_sum < 1e-300) Q_sum = 1e-300;

        // Compute gradients
        for (int i = 0; i < n; i++)
            std::fill(grad[i].begin(), grad[i].end(), 0.0);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                double q_ij = Q_num[i][j] / Q_sum;
                double mult = 4.0 * (exaggeration * P[i][j] - q_ij) * Q_num[i][j];
                for (int d = 0; d < dims; d++) {
                    grad[i][d] += mult * (Y[i][d] - Y[j][d]);
                }
            }
        }

        // Update with momentum and adaptive gains
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dims; d++) {
                // Adaptive gain: increase if gradient and velocity disagree
                double prev_step = Y[i][d] - Y_prev[i][d];
                bool same_sign = (grad[i][d] > 0) == (prev_step > 0);
                gains[i][d] = same_sign
                    ? std::max(gains[i][d] * 0.8, 0.01)
                    : gains[i][d] + 0.2;

                double new_val = Y[i][d]
                    - learning_rate * gains[i][d] * grad[i][d]
                    + momentum * (Y[i][d] - Y_prev[i][d]);

                Y_prev[i][d] = Y[i][d];
                Y[i][d] = new_val;
            }
        }

        // Re-center
        std::vector<double> mean(dims, 0.0);
        for (int i = 0; i < n; i++)
            for (int d = 0; d < dims; d++)
                mean[d] += Y[i][d];
        for (int d = 0; d < dims; d++) mean[d] /= n;
        for (int i = 0; i < n; i++)
            for (int d = 0; d < dims; d++)
                Y[i][d] -= mean[d];
    }

    ProjectionResult result;
    result.dims = dims;
    result.method = "tsne";
    result.points = Y;
    return result;
}
