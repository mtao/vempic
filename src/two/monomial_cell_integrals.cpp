#include "vem/two/monomial_cell_integrals.hpp"

#include <spdlog/spdlog.h>

#include <iostream>
#include <mtao/eigen/stack.hpp>
#include <mtao/eigen/stl2eigen.hpp>
#include <mtao/geometry/triangle_monomial_integrals.hpp>

#include "vem/polynomials/utils.hpp"

namespace vem::two {
namespace internal {

template <typename Derived>
mtao::VecXd monomial_cell_integrals(const VEMMesh2 &mesh, int index,
                                    int max_degree,
                                    const Eigen::MatrixBase<Derived> &center,
                                    double scale = 1) {
    const auto &V = mesh.V;
    const auto &E = mesh.E;
    const auto &cell_boundaries = mesh.face_boundary_map.at(index);
    Eigen::VectorXd monomial_integrals((max_degree + 1) * (max_degree + 2) / 2);
    // spdlog::info("Making monomial integrals of size: {} from max degree {}",
    //             monomial_integrals.size(), max_degree);
    monomial_integrals.setZero();
    for (auto &&[eidx, sgn] : cell_boundaries) {
        auto e = E.col(eidx);
        mtao::Vec2d b = (V.col(e(sgn ? 1 : 0)) - center) / scale;
        mtao::Vec2d c = (V.col(e(sgn ? 0 : 1)) - center) / scale;

        auto v = mtao::geometry::triangle_monomial_integrals<double>(
            max_degree, mtao::Vec2d::Zero(), b, c);
        // if the area of a triangle is very small lets just ignore it
        //if (std::abs(v[0]) < 1e-8) {
        //    continue;
        //} else if (v[0] < 0) {
        //    v[0] = -v[0];
        //}
        monomial_integrals += mtao::eigen::stl2eigen(v);
    }
    return monomial_integrals * (scale * scale);
}

}  // namespace internal

mtao::VecXd monomial_cell_integrals(const VEMMesh2 &mesh, int index,
                                    int max_degree) {
    if (index >= mesh.cell_count()) {
        spdlog::warn(
            "monomial cell integrals called on invalid cell index {} of {}",
            index, mesh.cell_count());
        return {};
    }
    return internal::monomial_cell_integrals(mesh, index, max_degree,
                                             mesh.C.col(index));
}
mtao::VecXd monomial_cell_integrals(const VEMMesh2 &mesh, int index,
                                    int max_degree, const mtao::Vec2d &center) {
    return internal::monomial_cell_integrals(mesh, index, max_degree, center);
}

mtao::VecXd scaled_monomial_cell_integrals(const VEMMesh2 &mesh, int index,
                                           double scale, int max_degree) {
    if (index >= mesh.cell_count()) {
        spdlog::warn(
            "monomial cell integrals called on invalid cell index {} of {}",
            index, mesh.cell_count());
        return {};
    }
    return internal::monomial_cell_integrals(mesh, index, max_degree,
                                             mesh.C.col(index), scale);
}
mtao::VecXd scaled_monomial_cell_integrals(const VEMMesh2 &mesh, int index,
                                           double scale, int max_degree,
                                           const mtao::Vec2d &center) {
    return internal::monomial_cell_integrals(mesh, index, max_degree, center,
                                             scale);
}

}  // namespace vem
