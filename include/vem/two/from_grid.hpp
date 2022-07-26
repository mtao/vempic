#pragma once

#include <mtao/geometry/grid/staggered_grid.hpp>

#include "mesh.hpp"

namespace vem {

class GridVEMMesh2 : public VEMMesh2,
                     public mtao::geometry::grid::StaggeredGrid2d {
   public:
    using GridType = mtao::geometry::grid::StaggeredGrid2d;
    GridVEMMesh2(const GridType &grid);
    int get_cell(const mtao::Vec2d &p, int last_known = -1) const override;
    bool in_cell(const mtao::Vec2d &p, int cell_index) const override;
    std::set<int> boundary_edge_indices() const override;
    double dx() const override;
    double diameter(size_t cell_index) const override;
};

// generates VEMMeshes. by default the centers chosen are centroidal
GridVEMMesh2 from_grid(const mtao::geometry::grid::StaggeredGrid2d &g);

// generates VEMMeshes. by default the centers chosen are centroidal
GridVEMMesh2 from_grid(const Eigen::AlignedBox<double, 2> &bb, int nx, int ny);

}  // namespace vem
