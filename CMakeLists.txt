cmake_minimum_required(VERSION 3.20)
project(sift CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(sift
  src/stb_impl.cpp
  src/main.cpp
  src/hash/dHash.cpp
  src/hash/pHash.cpp
  src/hash/wHash.cpp
  src/hash/image.cpp
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
  src/core
)

target_compile_options(sift PRIVATE -Wall -Wextra -O2)
target_link_libraries(sift PRIVATE pthread m)
