#pragma once

#include <mtao/geometry/grid/staggered_grid.hpp>

#include "grid_mesh.hpp"

namespace vem::two {


// generates VEMMeshes. by default the centers chosen are centroidal
GridVEMMesh2 from_grid(const mtao::geometry::grid::StaggeredGrid2d &g);

// generates VEMMeshes. by default the centers chosen are centroidal
GridVEMMesh2 from_grid(const Eigen::AlignedBox<double, 2> &bb, int nx, int ny);

}  // namespace vem
