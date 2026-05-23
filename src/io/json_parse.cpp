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

// ── Helpers for parse_project_json / parse_cluster_json ─────────────────────

static size_t skip_ws(const std::string& s, size_t p) {
    while (p < s.size() && (s[p]==' '||s[p]=='\n'||s[p]=='\r'||s[p]=='\t')) p++;
    return p;
}

// Returns position after ':' for the first occurrence of "key" in json[from..]
static size_t find_key_colon(const std::string& json, const std::string& key, size_t from = 0) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle, from);
    if (pos == std::string::npos) return std::string::npos;
    pos = json.find(':', pos + needle.size());
    return (pos != std::string::npos) ? pos + 1 : std::string::npos;
}

// Find the closing bracket/brace matching json[pos] ('{' or '[').
static size_t find_matching(const std::string& json, size_t pos) {
    char open = json[pos], close = (open == '{') ? '}' : ']';
    int depth = 0;
    bool in_str = false, escaped = false;
    for (size_t i = pos; i < json.size(); i++) {
        if (escaped)                  { escaped = false; continue; }
        if (in_str && json[i] == '\\') { escaped = true;  continue; }
        if (json[i] == '"')            { in_str = !in_str; continue; }
        if (in_str) continue;
        if      (json[i] == open)  depth++;
        else if (json[i] == close) { if (--depth == 0) return i; }
    }
    return std::string::npos;
}

// Parse a JSON string value starting at '"' (pos), advances pos past closing '"'.
static std::string parse_json_str(const std::string& json, size_t& pos) {
    pos++; // skip opening "
    std::string r;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case '"':  r += '"';  break;
                case '\\': r += '\\'; break;
                case '/':  r += '/';  break;
                case 'n':  r += '\n'; break;
                case 'r':  r += '\r'; break;
                case 't':  r += '\t'; break;
                default:   r += json[pos]; break;
            }
        } else {
            r += json[pos];
        }
        pos++;
    }
    if (pos < json.size()) pos++; // skip closing "
    return r;
}

// Parse ["str1", "str2", ...] starting at '[', advances pos past ']'.
static std::vector<std::string> parse_str_arr(const std::string& json, size_t& pos) {
    std::vector<std::string> r;
    pos++; // skip [
    pos = skip_ws(json, pos);
    while (pos < json.size() && json[pos] != ']') {
        if (json[pos] == '"') r.push_back(parse_json_str(json, pos));
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ',') pos++;
        pos = skip_ws(json, pos);
    }
    if (pos < json.size()) pos++; // skip ]
    return r;
}

// Parse [1, 2, 3, ...] starting at '[', advances pos past ']'.
static std::vector<int> parse_int_arr(const std::string& json, size_t& pos) {
    std::vector<int> r;
    pos++; // skip [
    pos = skip_ws(json, pos);
    while (pos < json.size() && json[pos] != ']') {
        if (json[pos] == '-' || std::isdigit((unsigned char)json[pos])) {
            char* end;
            r.push_back((int)std::strtol(json.c_str() + pos, &end, 10));
            pos = (size_t)(end - json.c_str());
        } else { pos++; }
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ',') pos++;
        pos = skip_ws(json, pos);
    }
    if (pos < json.size()) pos++; // skip ]
    return r;
}

// Parse [1.0, 2.0, ...] starting at '[', advances pos past ']'.
static std::vector<double> parse_dbl_arr(const std::string& json, size_t& pos) {
    std::vector<double> r;
    pos++; // skip [
    pos = skip_ws(json, pos);
    while (pos < json.size() && json[pos] != ']') {
        if (json[pos] == '-' || std::isdigit((unsigned char)json[pos])) {
            char* end;
            r.push_back(std::strtod(json.c_str() + pos, &end));
            pos = (size_t)(end - json.c_str());
        } else { pos++; }
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ',') pos++;
        pos = skip_ws(json, pos);
    }
    if (pos < json.size()) pos++; // skip ]
    return r;
}

static double extract_double(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return 0.0;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return 0.0;
    pos++;
    while (pos < json.size() && (json[pos]==' '||json[pos]=='\t')) pos++;
    return std::stod(json.substr(pos));
}

// ── parse_project_json ───────────────────────────────────────────────────────

ParsedProjection parse_project_json(const std::string& json) {
    ParsedProjection r;
    r.algorithm = extract_string(json, "algorithm");
    r.hash_size = extract_int(json, "hash_size");
    r.hash_bits = extract_int(json, "hash_bits");
    r.method    = extract_string(json, "method");
    r.dims      = extract_int(json, "dims");

    // files
    size_t pos = find_key_colon(json, "files");
    if (pos != std::string::npos) {
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == '[')
            r.files = parse_str_arr(json, pos);
    }

    // points (array of arrays)
    pos = find_key_colon(json, "points");
    if (pos != std::string::npos) {
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == '[') {
            pos++; // outer [
            pos = skip_ws(json, pos);
            while (pos < json.size() && json[pos] != ']') {
                if (json[pos] == '[') {
                    r.points.push_back(parse_dbl_arr(json, pos));
                } else { pos++; }
                pos = skip_ws(json, pos);
                if (pos < json.size() && json[pos] == ',') pos++;
                pos = skip_ws(json, pos);
            }
        }
    }

    // variance_explained (PCA only, optional)
    pos = find_key_colon(json, "variance_explained");
    if (pos != std::string::npos) {
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == '[')
            r.variance_explained = parse_dbl_arr(json, pos);
    }

    return r;
}

// ── parse_cluster_json ───────────────────────────────────────────────────────

ParsedCluster parse_cluster_json(const std::string& json) {
    ParsedCluster r;
    r.algorithm = extract_string(json, "algorithm");
    r.hash_size = extract_int(json, "hash_size");
    r.hash_bits = extract_int(json, "hash_bits");
    r.method    = extract_string(json, "method");

    // params: capture the raw JSON object verbatim
    size_t params_pos = json.find("\"params\"");
    if (params_pos != std::string::npos) {
        size_t brace = json.find('{', params_pos);
        if (brace != std::string::npos) {
            size_t end = find_matching(json, brace);
            if (end != std::string::npos)
                r.raw_params = json.substr(brace, end - brace + 1);
        }
    }

    // files array
    size_t pos = find_key_colon(json, "files");
    if (pos != std::string::npos) {
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == '[')
            r.files = parse_str_arr(json, pos);
    }

    // distance_matrix (array of int arrays)
    pos = find_key_colon(json, "distance_matrix");
    if (pos != std::string::npos) {
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == '[') {
            pos++; // outer [
            pos = skip_ws(json, pos);
            while (pos < json.size() && json[pos] != ']') {
                if (json[pos] == '[') {
                    r.distance_matrix.push_back(parse_int_arr(json, pos));
                } else { pos++; }
                pos = skip_ws(json, pos);
                if (pos < json.size() && json[pos] == ',') pos++;
                pos = skip_ws(json, pos);
            }
        }
    }

    // groups
    pos = find_key_colon(json, "groups");
    if (pos != std::string::npos) {
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == '[') {
            pos++; // outer [
            pos = skip_ws(json, pos);
            while (pos < json.size() && json[pos] != ']') {
                if (json[pos] == '{') {
                    size_t obj_end = find_matching(json, pos);
                    if (obj_end != std::string::npos) {
                        std::string obj = json.substr(pos, obj_end - pos + 1);
                        GroupInfo g;
                        g.id                   = extract_int(obj, "id");
                        g.max_internal_distance = extract_int(obj, "max_internal_distance");
                        g.avg_internal_distance = extract_double(obj, "avg_internal_distance");
                        size_t mp = find_key_colon(obj, "members");
                        if (mp != std::string::npos) {
                            mp = skip_ws(obj, mp);
                            if (obj[mp] == '[') g.members = parse_int_arr(obj, mp);
                        }
                        r.groups.push_back(std::move(g));
                        pos = obj_end + 1;
                    } else { pos++; }
                } else { pos++; }
                pos = skip_ws(json, pos);
                if (pos < json.size() && json[pos] == ',') pos++;
                pos = skip_ws(json, pos);
            }
        }
    }

    // ungrouped
    pos = find_key_colon(json, "ungrouped");
    if (pos != std::string::npos) {
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == '[')
            r.ungrouped = parse_int_arr(json, pos);
    }

    // membership (hdbscan only)
    size_t mem_key = json.find("\"membership\"");
    if (mem_key != std::string::npos) {
        r.has_membership = true;
        pos = find_key_colon(json, "membership");
        if (pos != std::string::npos) {
            pos = skip_ws(json, pos);
            if (pos < json.size() && json[pos] == '[') {
                pos++; // outer [
                pos = skip_ws(json, pos);
                while (pos < json.size() && json[pos] != ']') {
                    if (json[pos] == '{') {
                        size_t obj_end = find_matching(json, pos);
                        if (obj_end != std::string::npos) {
                            std::string obj = json.substr(pos, obj_end - pos + 1);
                            MembershipInfo m;
                            m.file       = extract_int(obj, "file");
                            m.group      = extract_int(obj, "group");
                            m.confidence = extract_double(obj, "confidence");
                            r.membership.push_back(m);
                            pos = obj_end + 1;
                        } else { pos++; }
                    } else { pos++; }
                    pos = skip_ws(json, pos);
                    if (pos < json.size() && json[pos] == ',') pos++;
                    pos = skip_ws(json, pos);
                }
            }
        }
    }

    return r;
}

} // namespace io
