
#include "vem/polynomial_partial_derivative.hpp"

#include <spdlog/spdlog.h>

#include "vem/polynomial_utils.hpp"

namespace vem::polynomials {
namespace two {
Eigen::SparseMatrix<double> partial_derivative(size_t max_degree, size_t axis) {
    size_t size = num_monomials_upto(max_degree);
    Eigen::SparseMatrix<double> R(size, size);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(size);
    for (size_t col = 0; col < size; ++col) {
        auto tup = index_to_exponents(col);
        auto &axis_val = tup[axis];
        if (axis_val == 0) {
            continue;
        }

        double val = axis_val;

        axis_val--;
        auto &&[xexp, yexp] = tup;

        size_t row = exponents_to_index(xexp, yexp);
        trips.emplace_back(row, col, val);
    }
    R.setFromTriplets(trips.begin(), trips.end());

    return R;
}
std::tuple<double, int> partial_derivative_single_index(size_t index,
                                                        size_t axis) {
    auto tup = index_to_exponents(index);
    auto &axis_val = tup[axis];
    if (axis_val <= 0) {
        return {0, -1};
    }
    double coeff = axis_val;

    axis_val--;
    auto &&[xexp, yexp] = tup;

    return {coeff, exponents_to_index(xexp, yexp)};
}
}  // namespace two
namespace three {
Eigen::SparseMatrix<double> partial_derivative(size_t max_degree, size_t axis) {
    size_t size = num_monomials_upto(max_degree);
    Eigen::SparseMatrix<double> R(size, size);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(size);
    for (size_t col = 0; col < size; ++col) {
        auto tup = index_to_exponents(col);
        auto &axis_val = tup[axis];
        if (axis_val == 0) {
            continue;
        }

        double val = axis_val;

        axis_val--;
        auto &&[xexp, yexp, zexp] = tup;

        size_t row = exponents_to_index(xexp, yexp, zexp);
        trips.emplace_back(row, col, val);
    }
    R.setFromTriplets(trips.begin(), trips.end());

    return R;
}
std::tuple<double, int> partial_derivative_single_index(size_t index,
                                                        size_t axis) {
    auto tup = index_to_exponents(index);
    auto &axis_val = tup[axis];
    if (axis_val <= 0) {
        return {0, -1};
    }
    double coeff = axis_val;

    axis_val--;
    auto &&[xexp, yexp, zexp] = tup;

    return {coeff, exponents_to_index(xexp, yexp, zexp)};
}
}  // namespace three
}  // namespace vem::polynomials
