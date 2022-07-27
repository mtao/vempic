#include "vem/two/triangle_mesh.hpp"

#include <mtao/geometry/circumcenter.h>
#include <mtao/geometry/mesh/boundary_facets.h>
#include <mtao/geometry/mesh/boundary_matrix.h>

namespace vem::two {

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


}  // namespace vem
