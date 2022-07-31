#pragma once
#include <map>
#include <set>
#include <vector>
namespace vem::utils {

std::vector<std::set<int>> face_neighboring_cells(
    const std::vector<std::map<int, bool>>& cells, size_t face_count);
}  // namespace vem::utils
