
#pragma once
#include "mesh.hpp"

namespace vem::three{

std::set<size_t> face_boundary_vertices(const VEMMesh3 &mesh, int face_index);
}  // namespace vem::utils
