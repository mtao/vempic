
#pragma once
#include "mesh.hpp"
namespace vem::three {
std::vector<std::set<int>> face_neighboring_cells(const VEMMesh3& mesh);
}  // namespace vem::three
