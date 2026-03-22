#pragma once

#include <cstdint>
#include <string>

// Returns a 64-bit perceptual hash of the image at the given path.
// Returns 0 on failure.

uint64_t dHash(const std::string& path);
uint64_t pHash(const std::string& path);
uint64_t wHash(const std::string& path);
