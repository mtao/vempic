#include <vem/two/from_grid.hpp>

namespace vem::two {


GridVEMMesh2 from_grid(const mtao::geometry::grid::StaggeredGrid2d &g) {
    return GridVEMMesh2(g);
}

// generates VEMMeshes. by default the centers chosen are centroidal
GridVEMMesh2 from_grid(const Eigen::AlignedBox<double, 2> &bb, int nx, int ny) {
    mtao::geometry::grid::StaggeredGrid2d g =
        mtao::geometry::grid::StaggeredGrid2d::from_bbox(
            bb, std::array<int, 2>{{nx, ny}});
    return from_grid(g);
}

}  // namespace vem
