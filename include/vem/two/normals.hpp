#pragma once
#include "mesh.hpp"

namespace vem::two {
mtao::ColVecs2d normals(const VEMMesh2 &mesh);
mtao::Vec2d normal(const VEMMesh2 &mesh, size_t edge_index);
}// namespace vem
