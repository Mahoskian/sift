#include "hash.hpp"
#include "cluster.hpp"
#include "io.hpp"
#include "threadpool.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <getopt.h>
#include <thread>

using HashFn = std::function<HashResult(const std::string&, int)>;

// ─── Usage ──────────────────────────────────────────────────────────────────

static void print_usage() {
    std::cerr <<
        "Usage:\n"
        "  sift hash <dir> [options]       Compute perceptual hashes\n"
        "  sift cluster <file> [options]    Cluster hashes by similarity\n"
        "\n"
        "Hash options:\n"
        "  --algo=<dhash|phash|whash>       Hash algorithm (default: dhash)\n"
        "  --size=<N>                       Hash grid size NxN (default: 8)\n"
        "  --threads=<N>                    Thread count (default: CPU cores)\n"
        "  --output=<file>                  Output JSON file (default: stdout)\n"
        "\n"
        "Cluster options:\n"
        "  --method=<threshold|hierarchical|hdbscan>\n"
        "                                   Clustering method (default: threshold)\n"
        "  --threshold=<N>                  Max hamming distance (threshold method, default: 10)\n"
        "  --linkage=<single|complete|average>\n"
        "                                   Linkage type (hierarchical, default: complete)\n"
        "  --cut-height=<N>                 Dendrogram cut height (hierarchical, default: 10)\n"
        "  --min-group=<N>                  Min cluster size (hdbscan, default: 3)\n"
        "  --threads=<N>                    Thread count (default: CPU cores)\n"
        "  --output=<file>                  Output JSON file (default: stdout)\n"
        "\n"
        "Examples:\n"
        "  sift hash ./photos --algo=phash --size=16 --output=hashes.json\n"
        "  sift cluster hashes.json --method=threshold --threshold=10\n"
        "  sift cluster hashes.json --method=hierarchical --linkage=complete --cut-height=15\n"
        "  sift cluster hashes.json --method=hdbscan --min-group=3\n"
        "  sift hash ./photos | sift cluster - --threshold=5\n";
}

// ─── JSON output: hash ──────────────────────────────────────────────────────

static void write_hash_json(
    std::ostream& out,
    const std::string& algo,
    int size,
    const std::vector<std::pair<std::string, HashResult>>& results
) {
    out << "{\n";
    out << "  \"algorithm\": \"" << algo << "\",\n";
    out << "  \"hash_size\": " << size << ",\n";
    out << "  \"hash_bits\": " << (size * size) << ",\n";
    out << "  \"files\": {\n";
    for (size_t i = 0; i < results.size(); i++) {
        const auto& [path, hash] = results[i];
        out << "    \"" << path << "\": \"" << hash.to_hex() << "\"";
        if (i + 1 < results.size()) out << ",";
        out << "\n";
    }
    out << "  }\n";
    out << "}\n";
}

// ─── JSON output: cluster ───────────────────────────────────────────────────

static void write_groups_json(
    std::ostream& out,
    const std::vector<GroupInfo>& groups,
    const std::vector<std::string>& files,
    int min_group_filter
) {
    // Separate groups from ungrouped (singletons below min_group_filter)
    std::vector<const GroupInfo*> real_groups;
    std::vector<int> ungrouped;

    for (auto& g : groups) {
        if ((int)g.members.size() >= min_group_filter) {
            real_groups.push_back(&g);
        } else {
            for (int m : g.members) ungrouped.push_back(m);
        }
    }

    out << "  \"groups\": [\n";
    for (size_t i = 0; i < real_groups.size(); i++) {
        auto* g = real_groups[i];
        out << "    {\n";
        out << "      \"id\": " << g->id << ",\n";
        out << "      \"members\": [";
        for (size_t j = 0; j < g->members.size(); j++) {
            out << g->members[j];
            if (j + 1 < g->members.size()) out << ", ";
        }
        out << "],\n";
        out << "      \"member_files\": [";
        for (size_t j = 0; j < g->members.size(); j++) {
            out << "\"" << files[g->members[j]] << "\"";
            if (j + 1 < g->members.size()) out << ", ";
        }
        out << "],\n";
        out << "      \"max_internal_distance\": " << g->max_internal_distance << ",\n";
        out << "      \"avg_internal_distance\": " << g->avg_internal_distance << "\n";
        out << "    }";
        if (i + 1 < real_groups.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"ungrouped\": [";
    for (size_t i = 0; i < ungrouped.size(); i++) {
        out << ungrouped[i];
        if (i + 1 < ungrouped.size()) out << ", ";
    }
    out << "]";
}

static void write_distance_matrix_json(std::ostream& out, const DistanceMatrix& dm) {
    out << "  \"distance_matrix\": [\n";
    for (int i = 0; i < dm.n; i++) {
        out << "    [";
        for (int j = 0; j < dm.n; j++) {
            out << dm.get(i, j);
            if (j + 1 < dm.n) out << ", ";
        }
        out << "]";
        if (i + 1 < dm.n) out << ",";
        out << "\n";
    }
    out << "  ]";
}

static void write_stats_json(std::ostream& out, const DistanceMatrix& dm,
                              const std::vector<GroupInfo>& groups,
                              const std::vector<int>& ungrouped,
                              int min_group_filter) {
    int total_groups = 0;
    int total_ungrouped = (int)ungrouped.size();
    for (auto& g : groups) {
        if ((int)g.members.size() >= min_group_filter)
            total_groups++;
        else
            total_ungrouped += (int)g.members.size();
    }

    out << "  \"stats\": {\n";
    out << "    \"total_files\": " << dm.n << ",\n";
    out << "    \"total_groups\": " << total_groups << ",\n";
    out << "    \"total_ungrouped\": " << total_ungrouped << ",\n";
    out << "    \"min_distance\": " << dm.min_distance() << ",\n";
    out << "    \"max_distance\": " << dm.max_distance() << ",\n";
    out << "    \"avg_distance\": " << dm.avg_distance() << "\n";
    out << "  }";
}

static void write_cluster_json(
    std::ostream& out,
    const ClusterInput& input,
    const DistanceMatrix& dm,
    const std::string& method,
    const std::vector<GroupInfo>& groups,
    int min_group_filter,
    // Method-specific data (nullptr if not applicable)
    const std::vector<DendrogramStep>* dendrogram,
    const std::vector<MembershipInfo>* membership,
    const std::vector<CondensedTreeNode>* condensed_tree,
    // Params
    const std::vector<std::pair<std::string, std::string>>& params
) {
    out << "{\n";
    out << "  \"algorithm\": \"" << input.algorithm << "\",\n";
    out << "  \"hash_size\": " << input.hash_size << ",\n";
    out << "  \"hash_bits\": " << input.hash_bits << ",\n";
    out << "  \"method\": \"" << method << "\",\n";

    // Params
    out << "  \"params\": {\n";
    for (size_t i = 0; i < params.size(); i++) {
        out << "    \"" << params[i].first << "\": " << params[i].second;
        if (i + 1 < params.size()) out << ",";
        out << "\n";
    }
    out << "  },\n";

    // Files array (ordered, indices used elsewhere)
    out << "  \"files\": [";
    for (size_t i = 0; i < input.files.size(); i++) {
        out << "\"" << input.files[i] << "\"";
        if (i + 1 < input.files.size()) out << ", ";
    }
    out << "],\n";

    // Distance matrix
    write_distance_matrix_json(out, dm);
    out << ",\n";

    // Dendrogram (hierarchical only)
    if (dendrogram) {
        out << "  \"dendrogram\": [\n";
        for (size_t i = 0; i < dendrogram->size(); i++) {
            auto& d = (*dendrogram)[i];
            out << "    {\"step\": " << d.step
                << ", \"merged\": [" << d.merged_a << ", " << d.merged_b
                << "], \"distance\": " << d.distance << "}";
            if (i + 1 < dendrogram->size()) out << ",";
            out << "\n";
        }
        out << "  ],\n";
    }

    // Membership (hdbscan only)
    if (membership) {
        out << "  \"membership\": [\n";
        for (size_t i = 0; i < membership->size(); i++) {
            auto& m = (*membership)[i];
            out << "    {\"file\": " << m.file
                << ", \"group\": " << m.group
                << ", \"confidence\": " << m.confidence << "}";
            if (i + 1 < membership->size()) out << ",";
            out << "\n";
        }
        out << "  ],\n";
    }

    // Condensed tree (hdbscan only)
    if (condensed_tree) {
        out << "  \"condensed_tree\": [\n";
        for (size_t i = 0; i < condensed_tree->size(); i++) {
            auto& ct = (*condensed_tree)[i];
            out << "    {\"parent\": " << ct.parent
                << ", \"child\": " << ct.child
                << ", \"lambda\": " << ct.lambda
                << ", \"child_size\": " << ct.child_size << "}";
            if (i + 1 < condensed_tree->size()) out << ",";
            out << "\n";
        }
        out << "  ],\n";
    }

    // Groups
    // Collect ungrouped
    std::vector<int> ungrouped;
    for (auto& g : groups)
        if ((int)g.members.size() < min_group_filter)
            for (int m : g.members) ungrouped.push_back(m);

    write_groups_json(out, groups, input.files, min_group_filter);
    out << ",\n";

    // Stats
    write_stats_json(out, dm, groups, ungrouped, min_group_filter);
    out << "\n";

    out << "}\n";
}

// ─── Subcommand: hash ───────────────────────────────────────────────────────

static int cmd_hash(int argc, char* argv[]) {
    std::string input_dir;
    std::string algo = "dhash";
    int hash_size = 8;
    int num_threads = (int)std::thread::hardware_concurrency();
    std::string output_file;

    static struct option long_opts[] = {
        {"algo",    required_argument, 0, 'a'},
        {"size",    required_argument, 0, 's'},
        {"threads", required_argument, 0, 't'},
        {"output",  required_argument, 0, 'o'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    optind = 1;
    int opt;
    while ((opt = getopt_long(argc, argv, "a:s:t:o:h", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'a': algo = optarg; break;
            case 's': hash_size = std::stoi(optarg); break;
            case 't': num_threads = std::stoi(optarg); break;
            case 'o': output_file = optarg; break;
            case 'h': print_usage(); return 0;
            default:  print_usage(); return 1;
        }
    }

    if (optind < argc) input_dir = argv[optind];

    if (input_dir.empty()) {
        std::cerr << "sift: no input directory specified\n";
        return 1;
    }
    if (hash_size < 2 || hash_size > 1024) {
        std::cerr << "sift: hash size must be between 2 and 1024\n";
        return 1;
    }
    if (num_threads < 1) num_threads = 1;

    HashFn hash_fn;
    if (algo == "dhash")      hash_fn = dhash;
    else if (algo == "phash") hash_fn = phash;
    else if (algo == "whash") hash_fn = whash;
    else {
        std::cerr << "sift: unknown algorithm '" << algo << "'\n";
        return 1;
    }

    auto images = io::scan_images(input_dir);
    if (images.empty()) {
        std::cerr << "sift: no images found in " << input_dir << "\n";
        return 1;
    }

    std::cerr << "sift: found " << images.size() << " images, hashing with "
              << algo << " " << hash_size << "x" << hash_size
              << " (" << (hash_size * hash_size) << " bits)"
              << " using " << num_threads << " threads\n";

    auto t_start = std::chrono::steady_clock::now();

    std::vector<std::pair<std::string, HashResult>> results(images.size());
    {
        ThreadPool pool(num_threads);
        std::vector<std::future<void>> futures;
        for (size_t i = 0; i < images.size(); i++) {
            futures.push_back(pool.submit([&, i]() {
                std::string path_str = images[i].string();
                HashResult h = hash_fn(path_str, hash_size);
                results[i] = {path_str, std::move(h)};
            }));
        }
        for (auto& f : futures) f.get();
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cerr << "sift: hashed " << images.size() << " images in " << elapsed_ms << " ms\n";

    if (output_file.empty()) {
        write_hash_json(std::cout, algo, hash_size, results);
    } else {
        std::ofstream ofs(output_file);
        if (!ofs) { std::cerr << "sift: cannot open: " << output_file << "\n"; return 1; }
        write_hash_json(ofs, algo, hash_size, results);
        std::cerr << "sift: wrote " << output_file << "\n";
    }

    return 0;
}

// ─── Subcommand: cluster ────────────────────────────────────────────────────

static int cmd_cluster(int argc, char* argv[]) {
    std::string input_file;
    std::string method = "threshold";
    int threshold = 10;
    std::string linkage = "complete";
    int cut_height = 10;
    int min_group = 3;
    int min_group_filter = 2;
    int num_threads = (int)std::thread::hardware_concurrency();
    std::string output_file;

    static struct option long_opts[] = {
        {"method",     required_argument, 0, 'm'},
        {"threshold",  required_argument, 0, 'T'},
        {"linkage",    required_argument, 0, 'l'},
        {"cut-height", required_argument, 0, 'c'},
        {"min-group",  required_argument, 0, 'g'},
        {"min-filter", required_argument, 0, 'f'},
        {"threads",    required_argument, 0, 't'},
        {"output",     required_argument, 0, 'o'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    optind = 1;
    int opt;
    while ((opt = getopt_long(argc, argv, "m:T:l:c:g:f:t:o:h", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'm': method = optarg; break;
            case 'T': threshold = std::stoi(optarg); break;
            case 'l': linkage = optarg; break;
            case 'c': cut_height = std::stoi(optarg); break;
            case 'g': min_group = std::stoi(optarg); break;
            case 'f': min_group_filter = std::stoi(optarg); break;
            case 't': num_threads = std::stoi(optarg); break;
            case 'o': output_file = optarg; break;
            case 'h': print_usage(); return 0;
            default:  print_usage(); return 1;
        }
    }

    if (optind < argc) input_file = argv[optind];

    if (input_file.empty()) {
        std::cerr << "sift: no input hash file specified (use - for stdin)\n";
        return 1;
    }
    if (num_threads < 1) num_threads = 1;

    // Read and parse hash JSON
    std::cerr << "sift: reading hashes from "
              << (input_file == "-" ? "stdin" : input_file) << "\n";

    std::string json_str = io::read_file(input_file);
    ClusterInput input = io::parse_hash_json(json_str);

    int n = (int)input.files.size();
    std::cerr << "sift: loaded " << n << " hashes ("
              << input.algorithm << " " << input.hash_size << "x" << input.hash_size << ")\n";

    // Compute distance matrix
    auto t_start = std::chrono::steady_clock::now();
    auto dm = compute_distance_matrix(input.hashes, num_threads);
    auto t_dm = std::chrono::steady_clock::now();
    double dm_ms = std::chrono::duration<double, std::milli>(t_dm - t_start).count();
    std::cerr << "sift: computed " << n << "x" << n
              << " distance matrix in " << dm_ms << " ms\n";

    // Run clustering
    std::vector<GroupInfo> groups;
    std::vector<DendrogramStep>* dendrogram_ptr = nullptr;
    std::vector<MembershipInfo>* membership_ptr = nullptr;
    std::vector<CondensedTreeNode>* condensed_tree_ptr = nullptr;
    std::vector<std::pair<std::string, std::string>> params;

    HierarchicalResult hier_result;
    HdbscanResult hdbscan_result;

    if (method == "threshold") {
        groups = threshold_cluster(dm, threshold);
        params.push_back({"threshold", std::to_string(threshold)});

    } else if (method == "hierarchical") {
        hier_result = hierarchical_cluster(dm, cut_height, linkage);
        groups = hier_result.groups;
        dendrogram_ptr = &hier_result.dendrogram;
        params.push_back({"linkage", "\"" + linkage + "\""});
        params.push_back({"cut_height", std::to_string(cut_height)});

    } else if (method == "hdbscan") {
        hdbscan_result = hdbscan_cluster(dm, min_group);
        groups = hdbscan_result.groups;
        membership_ptr = &hdbscan_result.membership;
        condensed_tree_ptr = &hdbscan_result.condensed_tree;
        params.push_back({"min_cluster_size", std::to_string(min_group)});

    } else {
        std::cerr << "sift: unknown method '" << method << "'\n";
        return 1;
    }

    auto t_end = std::chrono::steady_clock::now();
    double cluster_ms = std::chrono::duration<double, std::milli>(t_end - t_dm).count();

    // Count actual groups (above min_group_filter)
    int real_groups = 0;
    for (auto& g : groups)
        if ((int)g.members.size() >= min_group_filter) real_groups++;

    std::cerr << "sift: clustered with " << method << " in " << cluster_ms << " ms → "
              << real_groups << " groups\n";

    // Output
    auto write_out = [&](std::ostream& out) {
        write_cluster_json(out, input, dm, method, groups, min_group_filter,
                           dendrogram_ptr, membership_ptr, condensed_tree_ptr, params);
    };

    if (output_file.empty()) {
        write_out(std::cout);
    } else {
        std::ofstream ofs(output_file);
        if (!ofs) { std::cerr << "sift: cannot open: " << output_file << "\n"; return 1; }
        write_out(ofs);
        std::cerr << "sift: wrote " << output_file << "\n";
    }

    return 0;
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string subcommand = argv[1];
    if (subcommand == "--help" || subcommand == "-h") {
        print_usage();
        return 0;
    }

    // Shift past subcommand
    argc--;
    argv++;

    if (subcommand == "hash")    return cmd_hash(argc, argv);
    if (subcommand == "cluster") return cmd_cluster(argc, argv);

    std::cerr << "sift: unknown command '" << subcommand << "'\n";
    print_usage();
    return 1;
}
