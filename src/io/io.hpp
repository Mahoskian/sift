#pragma once

#include "../cluster/cluster.hpp"

#include <filesystem>
#include <string>
#include <vector>

// Returns paths of all supported image files in the given directory.
std::vector<std::string> scanInput(const std::filesystem::path& inputDir);

// Moves grouped files into subdirectories of outputDir.
void writeGroups(const std::vector<Group>& groups, const std::filesystem::path& outputDir);
