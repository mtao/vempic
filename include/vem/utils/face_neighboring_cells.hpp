#pragma once
#include "vem/mesh.hpp"
namespace vem::utils {
std::vector<std::set<int>> face_neighboring_cells(
    const std::vector<std::map<int, bool>>& cells, size_t face_count);
std::vector<std::set<int>> face_neighboring_cells(
    const std::vector<std::map<int, bool>>& cells);
std::vector<std::set<int>> face_neighboring_cells(const VEMMesh2& mesh);
std::vector<std::set<int>> face_neighboring_cells(const VEMMesh3& mesh);
}  // namespace vem::utils
