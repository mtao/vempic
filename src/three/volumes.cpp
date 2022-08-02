#include "vem/three/volumes.hpp"

namespace vem::three {

double volume(const VEMMesh3 &mesh, size_t cell_index) {
    auto C = mesh.C.col(cell_index);
    double v = 0;
    for (auto &&[fidx, sgn] : mesh.cell_boundary_map.at(cell_index)) {
        auto F = mesh.triangulated_faces.at(fidx);
        for (int j = 0; j < F.cols(); ++j) {
            auto f = F.col(j);
            auto a = mesh.V.col(f(0)) - C;
            auto b = mesh.V.col(f(1)) - C;
            auto c = mesh.V.col(f(2)) - C;
            double val = (sgn ? -1 : 1) * a.dot(b.cross(c));
            v += val;
        }
    }
    return v / 6;
}
mtao::VecXd volumes(const VEMMesh3 &mesh) {
    mtao::VecXd R(mesh.cell_count());
    for (size_t j = 0; j < mesh.cell_count(); ++j) {
        R(j) = volume(mesh, j);
    }
    return R;
}
}  // namespace vem::three
