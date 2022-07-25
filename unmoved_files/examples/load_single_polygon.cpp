#include <fmt/format.h>

#include <iostream>
#include <mtao/iterator/enumerate.hpp>
#include <vem/from_polygons.hpp>
#include <vem/mesh.hpp>
#include <vem/polynomial_utils.hpp>
#include <vem/point_sample_indexer.hpp>

#include <vem/monomial_cell_integrals.hpp>
int main(int argc, char *argv[]) {
    mtao::vector<mtao::ColVecs2d> polys;
    auto &P = polys.emplace_back();
    P.resize(2, 4);
    P.col(0) << 0, 0;
    P.col(1) << 1, 0;
    P.col(2) << 1, 1;
    P.col(3) << 0, 1;

    auto vem = vem::from_polygons(polys);
    std::cout << "Vertices: \n"
              << vem.V.transpose() << std::endl;
    std::cout << "Centers: \n"
              << vem.C.transpose() << std::endl;
    std::cout << "Boundary: " << std::endl;
    for (auto &&[cell_index, cell_map] :
         mtao::iterator::enumerate(vem.face_boundary_map)) {
        fmt::print("cell_index({})", cell_index);
        for (auto &&[c, sgn] : cell_map) {
            fmt::print("{}=>{} ", c, sgn ? -1 : 1);
        }
        fmt::print("\n");
    }

    fmt::print("Lets show some integrals on this unit square\n");

    int max_degree = 4;
    namespace polynomials = vem::polynomials::two;
    auto integrals = vem::monomial_cell_integrals(vem, 0, max_degree);
    for (int j = 0; j < polynomials::num_monomials_upto(max_degree); ++j) {
        auto [xexp, yexp] = polynomials::index_to_exponents(j);

        fmt::print("Integral of x^{} y^{} is {}\n", xexp, yexp, integrals(j));
    }

    // two points, so we have 4 samples, accurate to 2*4-3 = 5th
    vem::PointSampleIndexer basis(vem, 1);

    fmt::print("Offsets: {}\n", fmt::join(basis.edge_offsets(), " => "));

    {
        auto p = basis.evaluate_coefficients(
          [](const mtao::Vec2d &p) { return p.x(); });
        std::cout << p.transpose() << std::endl;
    }
    {
        auto p = basis.evaluate_coefficients(
          [](const mtao::Vec2d &p) { return p.y(); });
        std::cout << p.transpose() << std::endl;
    }
    std::cout << std::endl;
    {
        auto p = basis.evaluate_vector_field(
          [](const mtao::Vec2d &p) { return p.array() + 1.; });
        std::cout << "Position vfield" << std::endl;
        std::cout << p << std::endl;

        std::cout << "Fluxes" << std::endl;
        auto F = basis.boundary_fluxes(p);
        std::cout << F.transpose() << std::endl;
    }
    for (int eidx = 0; eidx < vem.edge_count(); ++eidx) {
    }

    return 0;
}
