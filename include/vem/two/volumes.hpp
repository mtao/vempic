#pragma once
#include "mesh.hpp"

namespace vem::two{
double volume(const VEMMesh2 &mesh, size_t cell_index);
mtao::VecXd volumes(const VEMMesh2 &mesh);
}  // namespace vem::utils
