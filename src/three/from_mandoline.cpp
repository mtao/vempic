#include "vem/three/from_mandoline.hpp"

#include <mandoline/construction/construct.hpp>
#include <mandoline/operators/volume3.hpp>
#include <mtao/logging/stopwatch.hpp>

//#include "vem/utils/merge_cells.hpp"

namespace vem::three {


MandolineVEMMesh3 from_mandoline(const mandoline::CutCellMesh<3> &ccm) {
    spdlog::info("returning a mandovem");
    auto sw = mtao::logging::hierarchical_stopwatch("Mandoline2VEM");
    return MandolineVEMMesh3(ccm);
}
MandolineVEMMesh3 from_mandoline(const Eigen::AlignedBox<double, 3> &bb, int nx,
                                 int ny, int nz, const mtao::ColVecs3d &V,
                                 const mtao::ColVecs3i &F, int adaptive_level) {
    mtao::geometry::grid::StaggeredGrid3d g =
        mtao::geometry::grid::StaggeredGrid3d::from_bbox(
            bb, std::array<int, 3>{{nx, ny, nz}});

    spdlog::info("Calling mandoline from grid object");

    mandoline::CutCellMesh<3> ccm;
    {
        auto sw = mtao::logging::hierarchical_stopwatch("Mandoline");
        ccm = mandoline::construction::from_grid(V, F, g, adaptive_level);
    }
    spdlog::info("triangulating");
    {
        auto sw = mtao::logging::hierarchical_stopwatch(
            "Mandoline_triangulating_faces");
        ccm.triangulate_faces(false);
    }

    return from_mandoline(ccm);
}

}  // namespace vem
