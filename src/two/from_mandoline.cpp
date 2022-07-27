#include "vem/two/from_mandoline.hpp"

#include <mandoline/construction/face_collapser.hpp>
#include <mandoline/operators/region_boundaries2.hpp>
#include <mandoline/operators/volume2.hpp>
#include <mtao/geometry/volume.hpp>

#include "mandoline/construction/construct2.hpp"
#include "vem/utils/merge_cells.hpp"
namespace vem::two {

MandolineVEMMesh2 from_mandoline(const mandoline::CutCellMesh<2> &ccm,
                                 bool delaminate) {
    return MandolineVEMMesh2(ccm, .2, delaminate);
}
// VEMMesh3 from_mandoline(const mandoline::CutCellMesh<3>& ccm) { return
// {}; }

MandolineVEMMesh2 from_mandoline(const Eigen::AlignedBox<double, 2> &bb, int nx,
                                 int ny, const mtao::ColVecs2d &V,
                                 const mtao::ColVecs2i &E, bool delaminate) {
    mtao::geometry::grid::StaggeredGrid2d g =
        mtao::geometry::grid::StaggeredGrid2d::from_bbox(
            bb, std::array<int, 2>{{nx, ny}});

    auto ccm = mandoline::construction::from_grid(V, E, g);

    return from_mandoline(ccm, delaminate);
}
std::set<int> MandolineVEMMesh2::boundary_edge_indices() const {
    if (boundary_edges.empty()) {
        return mandoline::operators::region_boundaries(_ccm);
    } else {
        return boundary_edges;
        // return VEMMesh2::boundary_edge_indices();
    }
}
}  // namespace vem
