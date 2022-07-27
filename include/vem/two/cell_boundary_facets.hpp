
#pragma once
#include "mesh.hpp"

namespace vem::two {

std::set<size_t> cell_boundary_vertices(const VEMMesh2 &mesh, int cell_index);
}  // namespace vem::utils
