#include "vem/two/monomial_face_integrals.hpp"

#include <spdlog/spdlog.h>

#include <iostream>
#include <mtao/eigen/stack.hpp>
#include <mtao/eigen/stl2eigen.hpp>
#include <mtao/geometry/edge_monomial_integrals.hpp>

#include "vem/polynomials/utils.hpp"

namespace vem::two {
namespace internal {

template <typename Derived>
mtao::VecXd monomial_face_integrals(const VEMMesh2 &mesh, int index,
                                    int max_degree,
                                    const Eigen::MatrixBase<Derived> &center,
                                    double scale = 1) {
    const auto &V = mesh.V;
    const auto &E = mesh.E;
    auto e = E.col(index);
    mtao::Vec2d b = (V.col(e(0)) - center) / scale;
    mtao::Vec2d c = (V.col(e(1)) - center) / scale;

    auto v = mtao::geometry::edge_monomial_integrals<double>(max_degree, b, c);
    Eigen::VectorXd monomial_integrals = mtao::eigen::stl2eigen(v);
    return monomial_integrals * (scale);
}

}  // namespace internal

mtao::VecXd monomial_face_integrals(const VEMMesh2 &mesh, int cell_index,
                                    int index, int max_degree) {
    if (index >= mesh.edge_count()) {
        spdlog::warn(
            "monomial face integrals called on invalid face index {} of {}",
            index, mesh.edge_count());
        return {};
    }
    return internal::monomial_face_integrals(mesh, index, max_degree,
                                             mesh.C.col(cell_index));
}
mtao::VecXd monomial_face_integrals(const VEMMesh2 &mesh, int index,
                                    int max_degree, const mtao::Vec2d &center) {
    return internal::monomial_face_integrals(mesh, index, max_degree, center);
}

mtao::VecXd scaled_monomial_face_integrals(const VEMMesh2 &mesh, int cell_index,
                                           int index, double scale,
                                           int max_degree) {
    if (index >= mesh.edge_count()) {
        spdlog::warn(
            "monomial face integrals called on invalid face index {} of {}",
            index, mesh.edge_count());
        return {};
    }
    return internal::monomial_face_integrals(mesh, index, max_degree,
                                             mesh.C.col(cell_index), scale);
}
mtao::VecXd scaled_monomial_face_integrals(const VEMMesh2 &mesh, int index,
                                           double scale, int max_degree,
                                           const mtao::Vec2d &center) {
    return internal::monomial_face_integrals(mesh, index, max_degree, center,
                                             scale);
}

}  // namespace vem
