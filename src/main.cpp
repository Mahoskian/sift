#include "hash.hpp"
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

static void print_usage() {
    std::cerr <<
        "Usage: sift hash <dir> [options]\n"
        "\n"
        "Options:\n"
        "  --algo=<dhash|phash|whash>  Hash algorithm (default: dhash)\n"
        "  --size=<N>                  Hash grid size NxN (default: 8)\n"
        "  --threads=<N>               Thread count (default: CPU cores)\n"
        "  --output=<file>             Output JSON file (default: stdout)\n"
        "  --help                      Show this help\n"
        "\n"
        "Examples:\n"
        "  sift hash ./photos --algo=phash --size=16\n"
        "  sift hash ./photos --algo=dhash --size=256 --output=hashes.json\n";
}

static void write_json(
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

    if (subcommand != "hash") {
        std::cerr << "sift: unknown command '" << subcommand << "'\n";
        print_usage();
        return 1;
    }

    // Shift argv past subcommand for getopt
    argc--;
    argv++;

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

    optind = 1;  // reset getopt
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

    // Remaining non-option argument is the directory
    if (optind < argc) {
        input_dir = argv[optind];
    }

    if (input_dir.empty()) {
        std::cerr << "sift: no input directory specified\n";
        print_usage();
        return 1;
    }

    if (hash_size < 2 || hash_size > 1024) {
        std::cerr << "sift: hash size must be between 2 and 1024\n";
        return 1;
    }

    if (num_threads < 1) num_threads = 1;

    // Select hash function
    HashFn hash_fn;
    if (algo == "dhash") {
        hash_fn = dhash;
    } else if (algo == "phash") {
        hash_fn = phash;
    } else if (algo == "whash") {
        hash_fn = whash;
    } else {
        std::cerr << "sift: unknown algorithm '" << algo << "' (use dhash, phash, or whash)\n";
        return 1;
    }

    // Scan images
    auto images = io::scan_images(input_dir);
    if (images.empty()) {
        std::cerr << "sift: no images found in " << input_dir << "\n";
        return 1;
    }

    std::cerr << "sift: found " << images.size() << " images, hashing with "
              << algo << " " << hash_size << "x" << hash_size
              << " (" << (hash_size * hash_size) << " bits)"
              << " using " << num_threads << " threads\n";

    // Hash all images in parallel
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
    std::cerr << "sift: hashed " << images.size() << " images in "
              << elapsed_ms << " ms\n";

    // Output JSON
    if (output_file.empty()) {
        write_json(std::cout, algo, hash_size, results);
    } else {
        std::ofstream ofs(output_file);
        if (!ofs) {
            std::cerr << "sift: cannot open output file: " << output_file << "\n";
            return 1;
        }
        write_json(ofs, algo, hash_size, results);
        std::cerr << "sift: wrote " << output_file << "\n";
    }

    return 0;
}
