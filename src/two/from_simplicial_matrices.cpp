#include "vem/from_simplicial_matrices.hpp"

#include <mtao/geometry/circumcenter.h>
#include <mtao/geometry/mesh/boundary_facets.h>
#include <mtao/geometry/mesh/boundary_matrix.h>

namespace vem {

std::vector<mtao::ColVecs3i> TriangleVEMMesh2::triangulated_faces() const {
    std::vector<mtao::ColVecs3i> ret(F.cols());
    for (int j = 0; j < ret.size(); ++j) {
        ret[j] = F.col(j);
    }
    return ret;
}
TriangleVEMMesh2::TriangleVEMMesh2(const mtao::ColVecs2d &V,
                                   const mtao::ColVecs3i &F)
    : F(F) {
    this->V = V;
    this->C = mtao::geometry::circumcenters(V, F);
    this->E = mtao::geometry::mesh::boundary_facets(F);
    this->face_boundary_map.resize(F.cols());

    Eigen::SparseMatrix<double> BMat =
        mtao::geometry::mesh::boundary_matrix<double>(F, this->E);
    using II = typename Eigen::SparseMatrix<double>::InnerIterator;

    for (int o = 0; o < BMat.outerSize(); ++o) {
        for (II it(BMat, o); it; ++it) {
            this->face_boundary_map[it.col()][it.row()] = it.value() < 0;
        }
    }
    for (auto &&[cidx, mp] :
         mtao::iterator::enumerate(this->face_boundary_map)) {
        std::cout << cidx << ": ";
        for (auto &&[f, sgn] : mp) {
            std::cout << "(" << f << ":" << (sgn ? '-' : '+') << ")";
        }
        std::cout << std::endl;
    }
}

bool TriangleVEMMesh2::in_cell(const mtao::Vec2d &p, int cell_index) const {
    return false;
}
int TriangleVEMMesh2::get_cell(const mtao::Vec2d &p, int last_known) const {
    return -1;
}

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
