#pragma once
#include <Eigen/Sparse>

namespace vem::polynomials {
namespace two {

// returns a map between coeffients in poly space and the gradient polynomials
// as a stack of coefficients
Eigen::SparseMatrix<double> gradient(size_t max_degree);
std::array<std::tuple<double, int>, 2> gradient_single_index(size_t index);
std::vector<std::array<std::tuple<double, int>, 2>> gradients_as_tuples(
    size_t max_degree);

Eigen::SparseMatrix<double> laplacian(size_t max_degree);

}  // namespace two
namespace three {
// returns a map between coeffients in poly space and the gradient polynomials
// as a stack of coefficients
Eigen::SparseMatrix<double> gradient(size_t max_degree);
std::array<std::tuple<double, int>, 3> gradient_single_index(size_t index);
std::vector<std::array<std::tuple<double, int>, 3>> gradients_as_tuples(
    size_t max_degree);
Eigen::SparseMatrix<double> laplacian(size_t max_degree);
}  // namespace three

template <int D>
std::vector<std::array<std::tuple<double, int>, D>> gradients_as_tuples(
    size_t max_degree) {
    if constexpr (D == 2) {
        return two::gradients_as_tuples(max_degree);
    } else if constexpr (D == 3) {
        return three::gradients_as_tuples(max_degree);
    }
}

}  // namespace vem::polynomials
