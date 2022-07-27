#include "vem/two/from_triangle_mesh.hpp"

#include <mtao/geometry/circumcenter.h>
#include <mtao/geometry/mesh/boundary_facets.h>
#include <mtao/geometry/mesh/boundary_matrix.h>

namespace vem::two {


TriangleVEMMesh2 from_triangle_mesh(const mtao::ColVecs2d &V,
                                    const mtao::ColVecs3i &F) {
    return TriangleVEMMesh2(V, F);
}
// VEMMesh3 from_tetrahedral_mesh(const mtao::ColVecs3d& V,
//                               const mtao::ColVecs4i& F) {
//    VEMMesh3 vem = from_tetrahedral_mesh(F);
//    vem.C = mtao::geometry::circumcenters(V, F);
//    return vem;
//}
// VEMTopology3 from_tetrahedral_mesh(const mtao::ColVecs4i& F) {
//    VEMTopology3 vemtop;
//
//    // TODO: fill in the faces
//
//    // vemtop.E = mtao::geometry::mesh::boundary_facets(F);
//    vemtop.face_boundary_map.resize(F.cols());
//
//    // Eigen::SparseMatrix<double> BMat =
//    //    mtao::geometry::mesh::boundary_matrix<double>(F, vemtop.E);
//    // using II = typename Eigen::SparseMatrix<double>::InnerIterator;
//
//    // for (int o = 0; o < BMat.outerSize(); ++o) {
//    //    for (II it(BMat, o); it; ++it) {
//    //        vemtop.face_boundary_map[it.col()][it.row()] = it.value() < 0;
//    //    }
//    //}
//    return vemtop;
//}

}  // namespace vem
