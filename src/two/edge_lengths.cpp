#include "vem/edge_lengths.hpp"
#include <mtao/iterator/enumerate.hpp>

namespace vem {
mtao::VecXd edge_lengths(const VEMMesh2 &mesh) {
    mtao::VecXd L(mesh.edge_count());
    for (int j = 0; j < mesh.edge_count(); ++j) {
        L(j) = edge_length(mesh, j);
    }
    return L;
}
mtao::VecXd edge_lengths(const VEMMesh2 &mesh, int cell_index) {

    auto &&fbm = mesh.face_boundary_map.at(cell_index);
    mtao::VecXd L(fbm.size());
    for (auto &&[idx, pr] : mtao::iterator::enumerate(fbm)) {
        auto &&[eidx, sgn] = pr;
        L(idx) = edge_length(mesh, eidx);
    }
    return L;
}
double edge_length(const VEMMesh2 &mesh, int edge_index) {
    auto e = mesh.E.col(edge_index);
    auto a = mesh.V.col(e(0));
    auto b = mesh.V.col(e(1));
    return (a - b).norm();
}
}// namespace vem
