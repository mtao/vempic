#include "vem/monomial_edge_integrals.hpp"

#include <spdlog/spdlog.h>

#include <mtao/eigen/stack.hpp>
#include <mtao/eigen/stl2eigen.hpp>
#include <mtao/geometry/edge_monomial_integrals.hpp>

namespace vem {
namespace internal {
template <typename Derived>
std::vector<double> single_edge_monomial_edge_integrals(
    const VEMMesh2 &mesh, int cell_index, int edge_index, int max_degree,
    const Eigen::MatrixBase<Derived> &center, double scale = 1) {
    const auto &V = mesh.V;
    const auto &E = mesh.E;
    const auto sgn = mesh.face_boundary_map.at(cell_index).at(edge_index);
    Eigen::VectorXd monomial_integrals((max_degree + 1) * (max_degree + 2) / 2);
    monomial_integrals.setZero();
    auto e = E.col(edge_index);
    mtao::Vec2d b = (V.col(e(sgn ? 1 : 0)) - center) / scale;
    mtao::Vec2d c = (V.col(e(sgn ? 0 : 1)) - center) / scale;

    auto v = mtao::geometry::edge_monomial_integrals<double>(max_degree, b, c);
    mtao::eigen::stl2eigen(v) *= scale;

    return v;
}
template <typename Derived>
std::map<int, std::vector<double>> per_edge_monomial_edge_integrals(
    const VEMMesh2 &mesh, int index, int max_degree,
    const Eigen::MatrixBase<Derived> &center, double scale = 1) {
    std::map<int, std::vector<double>> vals;
    const auto &V = mesh.V;
    const auto &E = mesh.E;
    const auto &edge_boundaries = mesh.face_boundary_map.at(index);
    Eigen::VectorXd monomial_integrals((max_degree + 1) * (max_degree + 2) / 2);
    monomial_integrals.setZero();
    for (auto &&[eidx, sgn] : edge_boundaries) {
        auto e = E.col(eidx);
        mtao::Vec2d b = (V.col(e(sgn ? 1 : 0)) - center) / scale;
        mtao::Vec2d c = (V.col(e(sgn ? 0 : 1)) - center) / scale;

        auto v =
            mtao::geometry::edge_monomial_integrals<double>(max_degree, b, c);

        mtao::eigen::stl2eigen(v) *= scale;
        vals[eidx] = v;
    }
    return vals;
}

template <typename Derived>
mtao::VecXd monomial_edge_integrals(const VEMMesh2 &mesh, int index,
                                    int max_degree,
                                    const Eigen::MatrixBase<Derived> &center,
                                    double scale = 1) {
    const auto &V = mesh.V;
    const auto &E = mesh.E;
    const auto &edge_boundaries = mesh.face_boundary_map.at(index);
    Eigen::VectorXd monomial_integrals((max_degree + 1) * (max_degree + 2) / 2);
    monomial_integrals.setZero();
    for (auto &&[eidx, sgn] : edge_boundaries) {
        auto e = E.col(eidx);
        mtao::Vec2d b = (V.col(e(sgn ? 1 : 0)) - center) / scale;
        mtao::Vec2d c = (V.col(e(sgn ? 0 : 1)) - center) / scale;

        auto v =
            mtao::geometry::edge_monomial_integrals<double>(max_degree, b, c);
        monomial_integrals += mtao::eigen::stl2eigen(v);
    }
    return monomial_integrals * (scale);
}
}  // namespace internal

mtao::VecXd scaled_monomial_edge_integrals(const VEMMesh2 &mesh, int index,
                                           double scale, int max_degree) {
    if (index >= mesh.edge_count()) {
        spdlog::warn(
            "monomial edge integrals called on invalid edge index {} of {}",
            index, mesh.edge_count());
        return {};
    }
    return internal::monomial_edge_integrals(mesh, index, max_degree,
                                             mesh.C.col(index), scale);
}
mtao::VecXd scaled_monomial_edge_integrals(const VEMMesh2 &mesh, int index,
                                           double scale, int max_degree,
                                           const mtao::Vec2d &center) {
    return internal::monomial_edge_integrals(mesh, index, max_degree, center,
                                             scale);
}

std::map<int, std::vector<double>> per_edge_scaled_monomial_edge_integrals(
    const VEMMesh2 &mesh, int index, double scale, int max_degree) {
    if (index >= mesh.edge_count()) {
        spdlog::warn(
            "monomial edge integrals called on invalid edge index {} of {}",
            index, mesh.edge_count());
        return {};
    }
    return internal::per_edge_monomial_edge_integrals(mesh, index, max_degree,
                                                      mesh.C.col(index), scale);
}
std::map<int, std::vector<double>> per_edge_scaled_monomial_edge_integrals(
    const VEMMesh2 &mesh, int index, double scale, int max_degree,
    const mtao::Vec2d &center) {
    return internal::per_edge_monomial_edge_integrals(mesh, index, max_degree,
                                                      center, scale);
}
std::vector<double> single_edge_scaled_monomial_edge_integrals(
    const VEMMesh2 &mesh, int cell_index, int edge_index, double scale,
    int max_degree) {
    if (cell_index >= mesh.edge_count()) {
        spdlog::warn(
            "monomial edge integrals called on invalid edge index {} of {}",
            cell_index, mesh.edge_count());
        return {};
    }
    return internal::single_edge_monomial_edge_integrals(
        mesh, cell_index, edge_index, max_degree, mesh.C.col(cell_index),
        scale);
}
std::vector<double> single_edge_scaled_monomial_edge_integrals(
    const VEMMesh2 &mesh, int cell_index, int edge_index, double scale,
    int max_degree, const mtao::Vec2d &center) {
    return internal::single_edge_monomial_edge_integrals(
        mesh, cell_index, edge_index, max_degree, center, scale);
}
}  // namespace vem
