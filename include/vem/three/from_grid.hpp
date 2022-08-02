#pragma once

#include <mtao/geometry/grid/staggered_grid.hpp>

#include "grid_mesh.hpp"
#include "mesh.hpp"

namespace vem::three {

// generates VEMMeshes. by default the centers chosen are centroidal
GridVEMMesh3 from_grid(const mtao::geometry::grid::StaggeredGrid3d &g);

// generates VEMMeshes. by default the centers chosen are centroidal
GridVEMMesh3 from_grid(const Eigen::AlignedBox<double, 3> &bb, int nx, int ny,
                       int nz);

}  // namespace vem::three
