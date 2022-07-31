#include <vem/normals.hpp>
#include <vem/utils/boundary_facets.hpp>

#include "vem/poisson_2d/constraints.hpp"

namespace vem::poisson_2d {

// constructs dirichlet boundary conditions for a linear function
ScalarConstraints linear_function_dirichlet(const VEMMesh2 &mesh,
                                            const double constant,
                                            const mtao::Vec2d &linear) {
    ScalarConstraints con;
    auto &pd = con.pointwise_dirichlet;
    auto Es = mesh.boundary_edge_indices();
    std::set<int> boundary_vertices;
    for (auto &&eidx : Es) {
        auto e = mesh.E.col(eidx);
        boundary_vertices.emplace(e(0));
        boundary_vertices.emplace(e(1));
    }
    for (auto &&vidx : boundary_vertices) {
        pd[vidx] = mesh.V.col(vidx).dot(linear) + constant;
    }
    return con;
}

// constructs neumann boundary conditions for a linear function, with the mean
// value set to constant
ScalarConstraints linear_function_neumann(const VEMMesh2 &mesh,
                                          const double constant,
                                          const mtao::Vec2d &linear) {
    ScalarConstraints con;
    /*
    auto& pd = con.pointwise_neumann;
    auto boundary_vertices = vem::utils::boundary_vertices(mesh);
    for (auto&& vidx : boundary_vertices) {
        pd[vidx] = linear;
    }
    */

    auto &pd = con.edge_integrated_flux_neumann;
    auto Es = mesh.boundary_edge_indices();
    auto N = vem::normals(mesh);

    for (auto &&eidx : Es) {
        auto e = mesh.E.col(eidx);
        auto a = mesh.V.col(e(0));
        auto b = mesh.V.col(e(1));
        auto n = N.col(eidx);

        double value =
            linear.norm() == 0 ? 0 : (linear.dot(n) * (b - a).norm());
        spdlog::info("edge {} got constraint {}", eidx, value);
        con.edge_integrated_flux_neumann[eidx] = value;
    }

    con.mean_value = constant;
    return con;
}

// constructs 0 dirichlet boundary conditions on the boundary and 0 on the
// membrane
ScalarConstraints pulled_membrane(const VEMMesh2 &mesh, const size_t index,
                                  const double value) {
    ScalarConstraints con;
    auto &pd = con.pointwise_dirichlet;
    std::set<int> boundary_vertices;
    auto Es = mesh.boundary_edge_indices();
    for (auto &&eidx : Es) {
        auto e = mesh.E.col(eidx);
        boundary_vertices.emplace(e(0));
        boundary_vertices.emplace(e(1));
    }
    for (auto &&vidx : boundary_vertices) {
        pd[vidx] = 0;
    }
    pd[index] = value;
    return con;
}

ScalarConstraints neumann_from_boundary_function(
    const VEMMesh2 &mesh,
    const std::function<std::tuple<mtao::Vec2d, bool>(const mtao::Vec2d,
                                                      double)> &f,
    double t) {
    ScalarConstraints con;
    /*
    auto& pd = con.pointwise_neumann;
    auto boundary_vertices = vem::utils::boundary_vertices(mesh);
    for (auto&& vidx : boundary_vertices) {
        pd[vidx] = linear;
    }
    */

    auto &pd = con.edge_integrated_flux_neumann;
    auto Es = mesh.boundary_edge_indices();
    auto N = vem::normals(mesh);

    for (auto &&eidx : Es) {
        auto e = mesh.E.col(eidx);
        auto a = mesh.V.col(e(0));
        auto b = mesh.V.col(e(1));
        auto n = N.col(eidx);

        auto [vec, constrained] = f((a + b) / 2, t);
        if (constrained) {
            double value = (vec.array() == 0).all() == 0
                               ? 0
                               : (vec.dot(n) * (b - a).norm());
            spdlog::info("edge {} got constraint {}", eidx, value);
            con.edge_integrated_flux_neumann[eidx] = value;
        }
    }

    return con;
}
}  // namespace vem::poisson_2d
