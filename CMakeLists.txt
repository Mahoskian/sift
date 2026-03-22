cmake_minimum_required(VERSION 3.20)
project(sift CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(sift src/main.cpp)

target_include_directories(sift PRIVATE third_party)

target_compile_options(sift PRIVATE -Wall -Wextra -O2)
