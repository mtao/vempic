#include <fmt/format.h>

#include <iostream>
#include <mtao/iterator/enumerate.hpp>
#include <vem/from_grid.hpp>
#include <vem/poisson_2d/poisson_vem.hpp>

int main(int argc, char *argv[]) {
    Eigen::AlignedBox<double, 2> bb;
    bb.min().setConstant(0);
    bb.max().setConstant(1);

    auto vem = vem::from_grid(bb, 5, 5);
    vem::poisson_2d::PoissonVEM2 pois(vem, 1);

    auto L = pois.stiffness_matrix();
    std::cout << mtao::RowVecXd::Ones(L.cols()) * L.transpose() << std::endl;
    std::cout << mtao::RowVecXd::Ones(L.cols()) * L << std::endl;

    return 0;
}
