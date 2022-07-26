
#pragma once
#include "vem/mesh.hpp"

namespace vem::utils {

std::set<size_t> face_boundary_vertices(const VEMMesh3 &mesh, int face_index);
}  // namespace vem::utils
