#include "hash.hpp"

#include <cstdint>
#include <string>
#include <vector>

// dHash: difference hash.
// Resize image to (size+1) x size grayscale.
// For each row, compare adjacent pixels: bit=1 if left > right.
// Produces size*size bits.

HashResult dhash(const std::string& path, int size) {
    // Load and resize to (size+1) columns x size rows
    auto gray = load_grayscale_resized(path, size + 1, size);
    if (gray.empty()) return {{}, 0};

    int w = size + 1;
    int total_bits = size * size;
    int total_bytes = (total_bits + 7) / 8;

    HashResult result;
    result.size = size;
    result.bits.resize(total_bytes, 0);

    int bit_idx = 0;
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            uint8_t left  = gray[row * w + col];
            uint8_t right = gray[row * w + col + 1];
            if (left > right) {
                result.set_bit(bit_idx);
            }
            bit_idx++;
        }
    }

    return result;
}
