#pragma once
#include "mesh.hpp"

namespace vem::two{
mtao::Vec2d normal(const VEMMesh2 &mesh, size_t cell_index, size_t edge_index);
}
