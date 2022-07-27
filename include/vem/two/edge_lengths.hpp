#pragma once
#include "mesh.hpp"

namespace vem::two {
mtao::VecXd edge_lengths(const VEMMesh2 &mesh);
double edge_length(const VEMMesh2 &mesh, int edge_index);
mtao::VecXd edge_lengths(const VEMMesh2 &mesh, int cell_index);
}// namespace vem
