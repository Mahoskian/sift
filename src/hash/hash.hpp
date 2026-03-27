#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

// Variable-size perceptual hash.
// For an NxN hash, stores N*N bits packed into bytes (MSB first).
struct HashResult {
    std::vector<uint8_t> bits;  // ceil(N*N / 8) bytes
    int size;                   // N (the hash is NxN)

    int num_bits() const { return size * size; }
    int num_bytes() const { return (int)bits.size(); }

    // Get bit at position i (0-indexed, MSB order within each byte)
    bool get_bit(int i) const {
        return (bits[i / 8] >> (7 - (i % 8))) & 1;
    }

    // Set bit at position i
    void set_bit(int i) {
        bits[i / 8] |= (1 << (7 - (i % 8)));
    }

    // Hex string representation
    std::string to_hex() const;

    // Hamming distance between two hashes (must be same size)
    static int hamming(const HashResult& a, const HashResult& b);
};

// Load image, convert to grayscale, resize to (w x h).
// Returns empty vector on failure.
std::vector<uint8_t> load_grayscale_resized(const std::string& path, int w, int h);

// Hash functions: all take image path + hash size N, return NxN bit hash.
HashResult dhash(const std::string& path, int size);
HashResult phash(const std::string& path, int size);
HashResult whash(const std::string& path, int size);
