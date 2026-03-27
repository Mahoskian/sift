#include "io.hpp"
#include "hash.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

// Minimal JSON parser for our known hash output format.
// Not a general-purpose JSON parser — only handles the exact structure
// produced by `sift hash`.

namespace io {

// Extract string value: "key": "value"
static std::string extract_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";

    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";

    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";

    return json.substr(pos + 1, end - pos - 1);
}

// Extract int value: "key": 123
static int extract_int(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return 0;

    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return 0;

    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    return std::stoi(json.substr(pos));
}

// Parse hex string to HashResult
static HashResult hex_to_hash(const std::string& hex, int hash_size) {
    HashResult result;
    result.size = hash_size;
    result.bits.resize(hex.size() / 2, 0);

    for (size_t i = 0; i < hex.size(); i += 2) {
        unsigned int byte;
        std::sscanf(hex.c_str() + i, "%2x", &byte);
        result.bits[i / 2] = (uint8_t)byte;
    }

    return result;
}

ClusterInput parse_hash_json(const std::string& json) {
    ClusterInput input;

    input.algorithm = extract_string(json, "algorithm");
    input.hash_size = extract_int(json, "hash_size");
    input.hash_bits = extract_int(json, "hash_bits");

    // Parse "files": { "path": "hex", ... }
    size_t files_pos = json.find("\"files\"");
    if (files_pos == std::string::npos)
        throw std::runtime_error("no 'files' key in hash JSON");

    size_t brace = json.find('{', files_pos);
    if (brace == std::string::npos)
        throw std::runtime_error("malformed files section");

    // Parse key-value pairs inside the files object
    size_t pos = brace + 1;
    while (pos < json.size()) {
        // Find next key
        size_t key_start = json.find('"', pos);
        if (key_start == std::string::npos) break;

        // Check if we've hit the closing brace
        size_t next_brace = json.find('}', pos);
        if (next_brace != std::string::npos && next_brace < key_start) break;

        size_t key_end = json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;

        std::string path = json.substr(key_start + 1, key_end - key_start - 1);

        // Find value
        size_t val_start = json.find('"', key_end + 1);
        if (val_start == std::string::npos) break;

        size_t val_end = json.find('"', val_start + 1);
        if (val_end == std::string::npos) break;

        std::string hex = json.substr(val_start + 1, val_end - val_start - 1);

        input.files.push_back(path);
        input.hashes.push_back(hex_to_hash(hex, input.hash_size));

        pos = val_end + 1;
    }

    return input;
}

std::string read_file(const std::string& path) {
    if (path == "-") {
        // Read from stdin
        std::string content;
        char buf[4096];
        while (std::cin.read(buf, sizeof(buf)))
            content.append(buf, std::cin.gcount());
        content.append(buf, std::cin.gcount());
        return content;
    }

    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open file: " + path);

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    return content;
}

} // namespace io
