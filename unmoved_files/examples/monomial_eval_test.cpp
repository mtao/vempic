#include <fmt/format.h>

#include <iostream>
#include <mtao/eigen/stack.hpp>
#include <vem/from_grid.hpp>
#include <vem/mesh.hpp>
#include <vem/poisson_2d/poisson_vem.hpp>

using namespace vem;
int main(int argc, char *argv[]) {
    int max_degree = 1;
    Eigen::AlignedBox2d bb(mtao::Vec2d(0., 0.), mtao::Vec2d(1., 1.));
    auto mesh = vem::from_grid(bb, 3, 3);
    vem::poisson_2d::PoissonVEM2 pmesh(mesh, max_degree, max_degree - 1);

    std::cout << "Moment size: " << pmesh.moment_size() << std::endl;
    mtao::VecXd D = mesh.V.row(1).transpose();

    auto Pi = pmesh.sample_to_polynomial_projection_matrix();
    spdlog::info("Pi{}x{} D{}", Pi.rows(), Pi.cols(), D.rows());
    mtao::VecXd P = Pi * D;

    std::cout << "P rows: " << P.rows() << std::endl;
    std::cout << P.transpose() << std::endl;

    std::cout << Pi << std::endl;

    mtao::ColVecs2d V(2, 8);

    V.col(0) << -1, -1;
    V.col(1) << 1, -1;
    V.col(2) << -1, 1;
    V.col(3) << 1, 1;
    V.col(4) << -.5, -.5;
    V.col(5) << .5, -.5;
    V.col(6) << -.5, .5;
    V.col(7) << .5, .5;

    /*
    for (int j = 0; j < V.cols(); ++j) {
        mtao::Vec2d p = mtao::Vec2d::Random();
        p = V.col(j);
        */
    for (int k = 0; k < mesh.cell_count(); ++k) {
        auto c = pmesh.get_cell(k);
        auto p = c.center();
        auto m0 = c.monomial(0);
        auto m1 = c.monomial(1);
        auto m2 = c.monomial(2);
        std::cout << p.transpose() << ") ";
        std::cout << c.evaluate_monomials(p).transpose() << " | " << m0(p)
                  << " " << m1(p) << " " << m2(p) << " "
                  << " ====> " << c.evaluate_monomial_function_from_block(p, P)
                  << std::endl;
    }
    std::cout << "Gradient operator:\n"
              << pmesh.monomial_indexer().gradient() << std::endl;
    // std::cout << std::endl;
    //}
    return 0;
}
