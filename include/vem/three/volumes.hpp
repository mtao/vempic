
#pragma once
#include "mesh.hpp"

namespace vem::three{
double volume(const VEMMesh3 &mesh, size_t cell_index);
mtao::VecXd volumes(const VEMMesh3 &mesh);
}  // namespace vem::utils
