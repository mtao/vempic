
#pragma once
#include "mesh.hpp"

namespace vem::utils {

std::set<size_t> cell_boundary_vertices(const VEMMesh3 &mesh, int cell_index);
}  // namespace vem::utils