#include "vem/polynomials/utils.hpp"

#include <spdlog/spdlog.h>

#include <cmath>
namespace vem::polynomials {
namespace two {
size_t num_monomials(size_t degree) { return degree + 1; }

size_t num_monomials_upto(size_t degree) {
    // for each degree d we can pick 0...d i.e d+1 options for the first var,
    // second is forced by this \sum_{d=0}^degree d+1 = \sum_{d=1}^{degree+1} =
    // (degree+1)(degree+1+1)/2
    return (degree + 1) * (degree + 2) / 2;
}

size_t monomial_index_degree(size_t index) {
    size_t d = std::ceil(std::sqrt(index));
    // offset is d * (d + 1) / 2
    // the minimal d s.t (d * (d+1) / 2 < index
    // (d * (d+1) / 2 < index < (d+1)(d+2) / 2
    // (d^2+d) < 2index < (d^2+3d+2)
    // d^2 + d - 2index == 0 at
    // d = .5 ( -1 +- sqrt(1 + 8 * index) )
    return std::floor(.5 * (-1 + std::sqrt(1 + 8 * index)));
}

std::array<size_t, 2> index_to_exponents(size_t index) {
    std::array<size_t, 2> ret;
    size_t d = monomial_index_degree(index);
    size_t off = d * (d + 1) / 2;
    size_t j = index - off;
    return std::array<size_t, 2>{{d - j, j}};
}

size_t exponents_to_index(size_t a, size_t b) {
    size_t d = a + b;
    size_t offset = (d * (d + 1)) / 2;
    return offset + b;
}
}  // namespace two
namespace three {
size_t num_monomials(size_t degree) {
    return ((degree + 1) * (degree + 2)) / 2;
}

size_t num_monomials_upto(size_t degree) {
    // for each degree d we can pick 0...d i.e d+1 options for the first var,
    return ((degree + 1) * (degree + 2) * (degree + 3)) / 6;
}

size_t monomial_index_degree(size_t index) {
    auto single_cubic_solution = [&](double a, double b, double c,
                                     double d) -> double {
        // something something depressed polynomial something something cadano's
        // formula
        double Q = double(3 * a * c - b * b) / double(9 * a * a);
        double R = double(9 * a * b * c - 27 * a * a * d - 2 * b * b * b) /
                   double(54 * a * a * a);

        // some bits can be moved around by moving operations around, but never
        // succeeded
        double QR = Q * Q * Q + R * R;
        if (QR > 0) {
            double S = std::cbrt(R + std::sqrt(QR));
            double T = std::cbrt(R - std::sqrt(QR));

            double ret = S + T - b / double(3 * a);
            return ret;
        } else {
            return 0;
        }
    };
    double sol = single_cubic_solution(1, 3, 2, -6 * double(index));
    double rsol = std::round(sol);
    if (std::abs(sol - rsol) <
        1e-5) {  // TODO: this will fail for rather high polynomials
        if (size_t e2i = exponents_to_index(rsol, 0, 0); e2i == index) {
            return rsol;
        } else {
            return rsol - 1;
        }
    } else {
        return std::floor(sol);
    }
}

std::array<size_t, 3> index_to_exponents(size_t index) {
    if (index == 0) {
        return {{0, 0, 0}};
    } else {
        size_t degree = monomial_index_degree(index);
        size_t offset = num_monomials_upto(degree - 1);
        size_t no_offset = index - offset;

        auto [y, z] = two::index_to_exponents(no_offset);
        // spdlog::info(
        //    "Index {} has degree {}, leftover degree {}, so shoving to y^{} "
        //    "z^{}",
        //    index, degree, no_offset, y, z);
        return {{degree - y - z, y, z}};
    }
}

size_t exponents_to_index(size_t a, size_t b, size_t c) {
    size_t d = a + b + c;
    size_t offset = num_monomials_upto(d - 1);
    size_t not_a = d - a;
    size_t a_offset = two::num_monomials_upto(not_a - 1);
    return offset + a_offset + c;
}
}  // namespace three
}  // namespace vem::polynomials
