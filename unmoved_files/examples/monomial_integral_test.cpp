#include <fmt/format.h>

#include <iostream>
#include <mtao/quadrature/simpsons.hpp>
#include <vem/from_grid.hpp>
#include <vem/mesh.hpp>
#include <vem/poisson_2d/poisson_vem.hpp>

#include "vem/poisson_2d/poisson_vem_cell.hpp"

using namespace vem;
int main(int argc, char *argv[]) {
    int max_degree = 1;
    Eigen::AlignedBox2d bb(mtao::Vec2d(-1, -1), mtao::Vec2d(1, 1));
    auto mesh = vem::from_grid(bb, 3, 3);
    vem::poisson_2d::PoissonVEM2 pmesh(mesh, max_degree, max_degree - 1);

    mtao::VecXd D = mesh.V.row(1).transpose();

    auto Pi = pmesh.point_to_polynomial_projection_matrix();
    mtao::VecXd P = Pi * D;

    std::cout << "P rows: " << P.rows() << std::endl;
    std::cout << P.transpose() << std::endl;

    std::cout << Pi << std::endl;

    auto run = [&](const poisson_2d::PoissonVEM2Cell &cell) {
        Eigen::AlignedBox2d bb;
        for (auto &&ind : cell.vertices()) {
            bb.extend(mesh.V.col(ind));
        }
        std::array<double, 2> min, max;
        mtao::eigen::stl2eigen(min) = bb.min();
        mtao::eigen::stl2eigen(max) = bb.max();
        auto integrals = cell.monomial_indexer.monomial_integrals(
            cell.index, 2 * monomial_degree());

        for(size_t j = 0; j < cell.num_monomials(); ++j
    };
    run(pmesh.get_cell(0),

    for (int k = 0; k < mesh.cell_count(); ++k) {
        std::cout << c.evaluate_monomials(p).transpose() << " ====> "
                  << c.evaluate_monomial_function_from_block(p, P) << std::endl;
    }
}
return 0;
}
