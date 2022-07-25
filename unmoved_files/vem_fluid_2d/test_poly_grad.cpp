#include <fmt/format.h>

#include <iostream>

#include "pointwise_vem_mesh2.hpp"

int main(int argc, char* argv[]) {
    PointwiseVEMMesh2 vem;
    vem.cells.emplace_back();
    auto& cell = vem.cells.front();
    vem.centers.resize(2, 1);
    vem.centers.col(0) << 0., 0.;
    vem.order = 2;
    auto G = vem.poly_gradient();
    std::cout << G << std::endl;

    for (size_t poly_idx = 0; poly_idx < vem.coefficient_size(); ++poly_idx) {
        auto [a,b] = vem.monomial_index_to_powers(poly_idx);
        size_t c = vem.powers_to_monomial_index(a,b);
        fmt::print("{} => {},{} => {}\n", poly_idx,a,b,c);
    }
    for (size_t poly_idx = 0; poly_idx < vem.coefficient_size(); ++poly_idx) {
        fmt::print("Poly index: {}\n", poly_idx);
        mtao::VecXd C = mtao::VecXd::Unit(vem.coefficient_size(), poly_idx);
        mtao::VecXd coeffs(vem.coefficient_size());
        coeffs.setRandom();
        coeffs = C;

        mtao::VecXd GV = G * coeffs;
        std::cout << GV.transpose() << std::endl;

        double eps = 1e-5;
        mtao::Vec2d xy;
        xy.setConstant(.5);
        double x = xy.x();
        double y = xy.y();
        double dx =
            (vem.polynomial_eval(0, mtao::Vec2d(x + eps, y), coeffs)(0) -
             vem.polynomial_eval(0, mtao::Vec2d(x - eps, y), coeffs)(0)) /
            (2 * eps);
        double dy =
            (vem.polynomial_eval(0, mtao::Vec2d(x, y + eps), coeffs)(0) -
             vem.polynomial_eval(0, mtao::Vec2d(x, y - eps), coeffs)(0)) /
            (2 * eps);
        double ddx =
            vem.polynomial_eval(0, xy,
                                GV.head(vem.coefficient_size()).eval())(0);
        double ddy =
            vem.polynomial_eval(0, xy,
                                GV.tail(vem.coefficient_size()).eval())(0);
        std::cout << "Coeffs:\n";
        std::cout << GV.head(vem.coefficient_size()).transpose() << std::endl;
        std::cout << GV.tail(vem.coefficient_size()).transpose() << std::endl;


        fmt::print("{} {} | {} {}\n", dx, dy, ddx, ddy);
    }
    return 0;
}
