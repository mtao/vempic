#pragma once
#include "vem/mesh.hpp"

namespace vem::utils {
double volume(const VEMMesh2 &mesh, size_t cell_index);
mtao::VecXd volumes(const VEMMesh2 &mesh);
double volume(const VEMMesh3 &mesh, size_t cell_index);
mtao::VecXd volumes(const VEMMesh3 &mesh);
}  // namespace vem::utils
