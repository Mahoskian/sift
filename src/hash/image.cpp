#include "hash.hpp"

#include "stb_image.h"
#include "stb_image_resize2.h"

#include <cstdio>
#include <vector>

std::vector<uint8_t> load_grayscale_resized(const std::string& path, int w, int h) {
    int orig_w, orig_h, channels;
    unsigned char* data = stbi_load(path.c_str(), &orig_w, &orig_h, &channels, 0);
    if (!data) {
        std::fprintf(stderr, "sift: failed to load image: %s\n", path.c_str());
        return {};
    }

    // Convert to grayscale if needed
    std::vector<uint8_t> gray(orig_w * orig_h);
    if (channels == 1) {
        for (int i = 0; i < orig_w * orig_h; i++) {
            gray[i] = data[i];
        }
    } else {
        // Luminance: 0.299*R + 0.587*G + 0.114*B (integer approximation)
        for (int i = 0; i < orig_w * orig_h; i++) {
            int r = data[i * channels + 0];
            int g = data[i * channels + 1];
            int b = data[i * channels + 2];
            gray[i] = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
        }
    }
    stbi_image_free(data);

    // Resize to target dimensions
    std::vector<uint8_t> resized(w * h);
    stbir_resize_uint8_linear(
        gray.data(), orig_w, orig_h, 0,
        resized.data(), w, h, 0,
        (stbir_pixel_layout)1  // STBIR_1CHANNEL
    );

    return resized;
}

// HashResult utilities
std::string HashResult::to_hex() const {
    static const char hex_chars[] = "0123456789abcdef";
    std::string out;
    out.reserve(bits.size() * 2);
    for (uint8_t b : bits) {
        out.push_back(hex_chars[(b >> 4) & 0xF]);
        out.push_back(hex_chars[b & 0xF]);
    }
    return out;
}

int HashResult::hamming(const HashResult& a, const HashResult& b) {
    int dist = 0;
    int len = (int)std::min(a.bits.size(), b.bits.size());
    for (int i = 0; i < len; i++) {
        dist += __builtin_popcount(a.bits[i] ^ b.bits[i]);
    }
    return dist;
}
