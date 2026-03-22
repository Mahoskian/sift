cmake_minimum_required(VERSION 3.20)
project(sift CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(sift
  src/main.cpp
  src/hash/dHash.cpp
  src/hash/pHash.cpp
  src/hash/wHash.cpp
  src/cluster/thresh.cpp
  src/cluster/hdbscan.cpp
  src/io/scanner.cpp
  src/io/writer.cpp
)

target_include_directories(sift PRIVATE
  third_party
  src/hash
  src/cluster
  src/io
)

target_compile_options(sift PRIVATE -Wall -Wextra -O2)
