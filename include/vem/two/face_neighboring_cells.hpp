
#pragma once
#include "mesh.hpp"
namespace vem::two {
std::vector<std::set<int>> face_neighboring_cells(const VEMMesh2& mesh);
}  // namespace vem::two
