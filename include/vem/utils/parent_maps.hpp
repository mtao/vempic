#pragma once
#include "vem/mesh.hpp"

namespace vem::utils {

std::map<size_t, std::set<size_t>> vertex_faces(const VEMMesh2 &mesh);

std::map<size_t, std::map<size_t, bool>> edge_faces(const VEMMesh2 &mesh);
}// namespace vem::utils
