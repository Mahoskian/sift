#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include "io/io.hpp"

#include <filesystem>
#include <bitset>
#include <thread>
#include <execution>
#include <iostream>
#include <getopt.h>

using namespace std;

int main(int argc, char* argv[]) {
  std::string input_dir;

  static struct option long_opts[] = {
    {"input", required_argument, 0, 'i'},
    {0, 0, 0, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "i:", long_opts, nullptr)) != -1) {
    if (opt == 'i') input_dir = optarg;
  }

  if (input_dir.empty()) {
    std::cerr << "Usage: sift --input=<dir>\n";
    return 1;
  }

  auto images = io::scan_images(input_dir);

  for (const auto& p : images) {
    std::cout << p << "\n";
  }

  std::cout << "Found: " << images.size() << " Files for Processing" << "\n";

  return 0;
}