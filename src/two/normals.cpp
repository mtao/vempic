#include "vem/two/normals.hpp"
namespace vem::two {
mtao::ColVecs2d normals(const VEMMesh2 &mesh) {
    mtao::ColVecs2d R(2, mesh.edge_count());
    for (int edge_index = 0; edge_index < mesh.edge_count(); ++edge_index) {
        auto e = mesh.E.col(edge_index);
        auto a = mesh.V.col(e(0));
        auto b = mesh.V.col(e(1));

        auto ab = b - a;
        R.col(edge_index) << -ab.y(), ab.x();
    }
    return R;
}
mtao::Vec2d normal(const VEMMesh2 &mesh, size_t edge_index) {
    auto e = mesh.E.col(edge_index);
    auto a = mesh.V.col(e(0));
    auto b = mesh.V.col(e(1));

    auto ab = b - a;
    return mtao::Vec2d(-ab.y(), ab.x());
}
}// namespace vem
