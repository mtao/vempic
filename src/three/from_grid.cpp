
#include <mtao/geometry/grid/triangulation.hpp>
#include <vem/three/from_grid.hpp>

namespace vem::three {

GridVEMMesh3 from_grid(const mtao::geometry::grid::StaggeredGrid3d &g) {
    return GridVEMMesh3(g);
}

// generates VEMMeshes. by default the centers chosen are centroidal
GridVEMMesh3 from_grid(const Eigen::AlignedBox<double, 3> &bb, int nx, int ny,
                       int nz) {
    mtao::geometry::grid::StaggeredGrid3d g =
        mtao::geometry::grid::StaggeredGrid3d::from_bbox(
            bb, std::array<int, 3>{{nx, ny, nz}});
    std::cout << "Input bbox: " << bb.min().transpose() << ","
              << bb.max().transpose() << " with shape " << nx << "," << ny
              << "," << nz << std::endl;
    return from_grid(g);
}

}  // namespace vem::three
