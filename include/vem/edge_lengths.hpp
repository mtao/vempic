#pragma once
#include "vem/mesh.hpp"

namespace vem {
mtao::VecXd edge_lengths(const VEMMesh2 &mesh);
double edge_length(const VEMMesh2 &mesh, int edge_index);
mtao::VecXd edge_lengths(const VEMMesh2 &mesh, int cell_index);
}// namespace vem
