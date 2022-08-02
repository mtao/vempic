#include "vem/three/monomial_basis_indexer.hpp"

namespace vem::detail {
template<>
void MonomialBasisIndexer<2, 3>::fill_diameters() {

    _diameters.resize(_mesh.face_count());
    for (auto &&[idx, d] : mtao::iterator::enumerate(_diameters)) {
        auto c = _mesh.FC.col(idx);
        auto VI = three::face_boundary_vertices(mesh(), idx);
        d = 0;
        for (auto &&vi : VI) {
            if (vi < mesh().V.cols()) {
                double pr = (mesh().V.col(vi) - c).norm();
                if (pr > d) {
                    d = pr;
                }
            }
        }
        d *= 2;
    }
}
//template<>
//void MonomialBasisIndexer<1, 3>::fill_diameters() {
//
//    _diameters.resize(_mesh.edge_count());
//    for (auto &&[idx, d] : mtao::iterator::enumerate(_diameters)) {
//        d = _mesh.boundary_diameter(idx);
//        /*
//        auto c = _mesh.boundary_center(idx);
//        auto VI = utils::cell_boundary_vertices(mesh(), idx);
//        d = 0;
//        for (auto &&vi : VI) {
//            double pr = (mesh().V.col(vi) - c).norm();
//            if (pr > d) {
//                d = pr;
//            }
//        }
//        d *= 2;
//        */
//    }
//}
}// namespace vem
