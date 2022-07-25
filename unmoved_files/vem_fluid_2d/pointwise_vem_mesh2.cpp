#include "pointwise_vem_mesh2.hpp"

#include <spdlog/spdlog.h>

#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/triangle_monomial_integrals.hpp>
#include <mtao/iterator/enumerate.hpp>
#include <numeric>
Eigen::SparseMatrix<double> PointwiseVEMMesh2::divergence_sample2poly() const {
    Eigen::SparseMatrix<double> S2C = sample2cell_coefficients();

    // because the divergence is two dimensions stacked we need to create the
    // block matrix that undoes it
    Eigen::SparseMatrix<double> S2C2 =
        mtao::eigen::sparse_block_diagonal_repmats(S2C, 2);
    return divergence() * S2C2;
}
Eigen::SparseMatrix<double> PointwiseVEMMesh2::gradient_sample2poly() const {
    Eigen::SparseMatrix<double> S2C = sample2cell_coefficients();

    return gradient() * S2C;
}
Eigen::SparseMatrix<double> PointwiseVEMMesh2::divergence_sample2sample()
    const {
    return poly2sample() * divergence_sample2poly();
}
Eigen::SparseMatrix<double> PointwiseVEMMesh2::gradient_sample2sample() const {
    return mtao::eigen::sparse_block_diagonal_repmats(poly2sample(), 2) *
           gradient_sample2poly();
}
Eigen::SparseMatrix<double>
PointwiseVEMMesh2::integrated_divergence_sample2sample() const {
    auto G = gradient_sample2poly();
    auto I = monomial_integrals();
    auto S2C = sample2cell_coefficients();
    Eigen::SparseMatrix<double> S2C2 =
        mtao::eigen::sparse_block_diagonal_repmats(S2C, 2);
    mtao::VecXd I2 = mtao::eigen::vstack(I, I);
    return G.transpose() * I2.asDiagonal() * S2C2;
    return gradient_sample2poly().transpose() * poly2sample() *
           monomial_integrals().asDiagonal() * divergence_sample2poly();
}
Eigen::SparseMatrix<double>
PointwiseVEMMesh2::integrated_divergence_sample2adj_sample() const {
    auto S2C = sample2cell_coefficients();
    auto S2C2 = mtao::eigen::sparse_block_diagonal_repmats(S2C, 2);
    auto MI = monomial_integrals();
    auto D = divergence();
    spdlog::warn("{1}x{0} {2} {3}x{4} {5}x{6}", S2C.rows(), S2C.cols(),
                 MI.size(), D.rows(), D.cols(), S2C2.rows(), S2C2.cols());
    return S2C.transpose() * MI.asDiagonal() * D * S2C2;
}
Eigen::SparseMatrix<double>
PointwiseVEMMesh2::integrated_divergence_poly2adj_sample(
    const std::set<size_t>& disengaged_cells) const {
    auto S2C = sample2cell_coefficients();
    auto MI = monomial_integrals(disengaged_cells);
    auto MI2 = mtao::eigen::vstack(MI, MI);

    Eigen::SparseMatrix<double> D =
        S2C.transpose() * gradient().transpose() * MI2.asDiagonal();
    return D;
}
mtao::MatXd PointwiseVEMMesh2::laplacian_sample2sample(
    const std::set<size_t>& disengaged_cells) const {
    mtao::MatXd G = gradient_sample2poly();
    mtao::MatXd D = integrated_divergence_poly2adj_sample(disengaged_cells);
    return D * G + 100 * regression_error_bilinear(disengaged_cells);

    // auto S2C = sample2cell_coefficients();
    // return S2C.transpose() * laplacian(disengaged_cells) * S2C;
}
Eigen::SparseMatrix<double> PointwiseVEMMesh2::regression_error_bilinear(
    const std::set<size_t>& disengaged_cells) const {
    size_t off = coefficient_size();
    std::vector<Eigen::Triplet<double>> trips;

    Eigen::SparseMatrix<double> ret(num_samples(), num_samples());
    trips.reserve(num_cells() * num_samples());

    for (int j = 0; j < num_cells(); ++j) {
        if (disengaged_cells.find(j) != disengaged_cells.end()) {
            continue;
        }
        auto indices = cell_sample_indices_vec(j);
        auto [P, R] = poly_projection_sample2sample(j, indices);
        if (R.size() == 0) {
            continue;
        }
        mtao::MatXd N = R * R.transpose();
        spdlog::warn("R shape {} {}; N shape {} {}; indices {}", R.rows(),
                     R.cols(), N.rows(), N.cols(), indices.size());
        // std::cout << "percelllap\n" << L << std::endl;
        for (int k = 0; k < indices.size(); ++k) {
            for (int l = 0; l < indices.size(); ++l) {
                double val = N(k, l);
                if (val != 0) {
                    trips.emplace_back(indices.at(k), indices.at(l), val);
                }
            }
        }
    }
    ret.setFromTriplets(trips.begin(), trips.end());

    return ret;
}
/*
mtao::MatXd VEMMesh2::per_monomial_divergence(size_t index) const {
    mtao::MatXd M(2 * coefficient_size(), coefficient_size());

    auto monomial_integrals =
        per_cell_per_monomial_integral(index, 2 * order - 1);
    for (int d1 = 0; d1 <= order; ++d1) {
        int off1 = (d1 * (d1 + 1)) / 2;
        for (int j1 = 0; j1 < d1 + 1; ++j1) {
            for (int d2 = d1; d2 <= order; ++d2) {
                int off2 = (d2 * (d2 + 1)) / 2;
                for (int j2 = (d1 == d2) ? (j1) : 0; j2 < d2 + 1; ++j2) {
                    // \int \nabla (x^j y^k)  \cdot \nabla (x^l y^m
                    // = j * l \int x^{l+j-2} y^{k+m} + km x^{j+l} y^{k+m-2}
                    size_t j = j1;
                    size_t k = (d1 - j1);
                    size_t l = j2;
                    size_t m = (d2 - j2);
                    size_t d = d1 + d2 - 2;
                    // fmt::print("j{} + k{} = d{}; l{} + m{} = d{}; total order
                    // = {}\n", j,k,d1,l,m,d2, d);

                    size_t off = (d * (d + 1)) / 2;
                    double val = 0;
                    if (j >= 1 && l >= 1) {
                        val += j * l * monomial_integrals[off + j + l - 2];
                    }
                    if (k >= 1 && m >= 1) {
                        val += k * m * monomial_integrals[off + k + m - 2];
                    }
                    M(off2 + j2, off1 + j1) = M(off1 + j1, off2 + j2) = val;
                }
            }
        }
    }
    return M;
}
*/
std::array<mtao::MatXd, 2> PointwiseVEMMesh2::poly_projection_sample2sample(
    size_t index) const {
    auto indices = cell_sample_indices_vec(index);
    return poly_projection_sample2sample(index, indices);
}
mtao::MatXd PointwiseVEMMesh2::poly_projection_sample2poly(
    size_t index, const std::vector<size_t>& sample_indices) const {
    auto coeff_mat = poly_coefficient_matrix(index, sample_indices);
    mtao::MatXd m = coeff_mat.transpose() * coeff_mat;
    Eigen::SelfAdjointEigenSolver<mtao::MatXd> solver(m);
    size_t usable_eigenvals =
        std::min(coefficient_size(), sample_indices.size());
    mtao::VecXd invdiag =
        1.0 / solver.eigenvalues().tail(usable_eigenvals).array();

    mtao::MatXd B = solver.eigenvectors().leftCols(usable_eigenvals);

    return coeff_mat * B * invdiag.asDiagonal() * B.transpose();
}
mtao::MatXd PointwiseVEMMesh2::poly_projection_sample2poly(size_t index) const {
    auto indices = cell_sample_indices_vec(index);
    return poly_projection_sample2poly(index, indices);
}
mtao::MatXd PointwiseVEMMesh2::poly_projection_kernel(
    size_t index, const std::vector<size_t>& sample_indices) const {
    auto coeff_mat = poly_coefficient_matrix(index, sample_indices);
    mtao::MatXd m = coeff_mat.transpose() * coeff_mat;
    Eigen::SelfAdjointEigenSolver<mtao::MatXd> solver(m);
    size_t usable_eigenvals =
        std::min(coefficient_size(), sample_indices.size());
    mtao::VecXd invdiag =
        1.0 / solver.eigenvalues().tail(usable_eigenvals).array();

    if (m.cols() > usable_eigenvals) {
        mtao::MatXd N =
            solver.eigenvectors().rightCols(m.cols() - usable_eigenvals);
        return N;
    } else {
        return {};
    }

    mtao::MatXd B = solver.eigenvectors().leftCols(usable_eigenvals);

    return coeff_mat * B * invdiag.asDiagonal() * B.transpose();
}
mtao::MatXd PointwiseVEMMesh2::poly_projection_kernel(size_t index) const {
    auto indices = cell_sample_indices_vec(index);
    return poly_projection_kernel(index, indices);
}

Eigen::SparseMatrix<double> PointwiseVEMMesh2::poly_projection_sample2poly(
    const std::set<size_t>& disengaged_cells) const {
    size_t off = coefficient_size();
    std::vector<Eigen::Triplet<double>> trips;

    Eigen::SparseMatrix<double> ret(num_samples(), num_samples());
    trips.reserve(num_cells() * num_samples());

    for (int j = 0; j < num_cells(); ++j) {
        if (disengaged_cells.find(j) != disengaged_cells.end()) {
            continue;
        }
        auto indices = cell_sample_indices_vec(j);
        auto P = poly_projection_sample2poly(j, indices);
        // std::cout << "percelllap\n" << L << std::endl;
        int off = j * off;
        for (int k = 0; k < P.rows(); ++k) {
            for (int l = 0; l < P.cols(); ++l) {
                double val = P(k, l);
                if (val != 0) {
                    trips.emplace_back(off + k, indices.at(l), val);
                }
            }
        }
    }
    ret.setFromTriplets(trips.begin(), trips.end());

    return ret;
}
mtao::MatXd PointwiseVEMMesh2::per_cell_laplacian(size_t cell_index) const {
    auto P = poly_projection_sample2poly(cell_index);
    auto M = per_monomial_laplacian(cell_index);
    auto [BB, N] = poly_projection_sample2sample(cell_index);
    // std::cout << "Poly projection:\n" << P << std::endl;
    // std::cout << "Monomial lap:\n" << M << std::endl;
    if (N.size() > 0 || projection_lambda != 0) {
        return P.transpose() * M * P + projection_lambda * N * N.transpose();
    } else {
        return P.transpose() * M * P;
    }
}
/*
Eigen::SparseMatrix<double> PointwiseVEMMesh2::laplacian() const {
    size_t system_size = num_samples();
    mtao::MatXd LD(system_size, system_size);  // = L;
    LD.setZero();
    for (size_t cidx = 0; cidx < num_cells(); ++cidx) {
        if (disengaged_cells.find(cidx) != disengaged_cells.end()) {
            continue;
        }
        if (cells.at(cidx).size() > 0) {
            auto C = cell_sample_to_world_sample(cidx);
            auto PL = per_cell_laplacian(cidx);
            // std::cout << "Permutation:\n";
            // std::cout << C << std::endl;

            LD.topLeftCorner(num_samples(), num_samples()) +=
                C * PL * C.transpose();
        }
    }
    return LD;
}
*/
std::array<mtao::MatXd, 2> PointwiseVEMMesh2::poly_projection_sample2sample(
    size_t index, const std::vector<size_t>& sample_indices) const {
    auto coeff_mat = poly_coefficient_matrix(index, sample_indices);
    mtao::MatXd m = coeff_mat.transpose() * coeff_mat;
    Eigen::SelfAdjointEigenSolver<mtao::MatXd> solver(m);
    size_t usable_eigenvals =
        std::min(coefficient_size(), sample_indices.size());
    mtao::VecXd invdiag =
        1.0 / solver.eigenvalues().tail(usable_eigenvals).array();

    mtao::MatXd B = solver.eigenvectors().leftCols(usable_eigenvals);
    if (m.cols() > usable_eigenvals) {
        mtao::MatXd N =
            solver.eigenvectors().rightCols(m.cols() - usable_eigenvals);

        // std::cout << solver.eigenvectors() << std::endl;
        // std::cout << m << std::endl;
        return {B * B.transpose(), N};
    } else {
        return {B, {}};
    }
    // return {B * invdiag.asDiagonal() * B.transpose(), N};
}

std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd>
PointwiseVEMMesh2::orthogonal_neumann_entries(
    const std::map<size_t, double>& target_fluxes) const {
    if (target_fluxes.size() == 0) {
        return {};
    }
    // flatten the edge indices into a coherent order
    std::vector<size_t> edge_indices(target_fluxes.size());
    std::transform(target_fluxes.begin(), target_fluxes.end(),
                   edge_indices.begin(),
                   [](auto&& pr) { return std::get<0>(pr); });

    // get the number of indices for each segment
    std::vector<size_t> boundary_sizes(target_fluxes.size());
    std::transform(edge_indices.begin(), edge_indices.end(),
                   boundary_sizes.begin(),
                   std::bind(&VEMMeshBase::num_interior_samples_on_boundary,
                             dynamic_cast<const VEMMeshBase*>(this),
                             std::placeholders::_1));

    // get offsets for easy acccess
    std::vector<size_t> constraint_matrix_offsets(target_fluxes.size() + 1);
    std::partial_sum(edge_indices.begin(), edge_indices.end(),
                     constraint_matrix_offsets.begin() + 1);

    auto COB = coboundary();
    // for (auto it = COB.begin(); it != COB.end();) {
    //    if (target_fluxes.find(it->first) != target_fluxes.end()) {
    //        ++it;
    //    } else {
    //        COB.erase(it);
    //    }
    //}
    // std::map<size_t, mtao::MatXd> cell_projectors;
    // for (auto&& [eidx, pr] : COB) {
    //    for (auto&& a : pr) {
    //        if (!cell_projectors.contains(a)) {
    //            cell_projectors[a] = poly_projection_sample2poly(a);
    //        }
    //    }
    //}

    // first we construct a map from polynomials -> satisfaction of the neumann
    // constraint
    size_t size = constraint_matrix_offsets.back();
    Eigen::SparseMatrix<double> A(target_fluxes.size(),
                                  num_cells() * coefficient_size());
    std::vector<Eigen::Triplet<double>> trips;

    for (auto&& [local_index, edge_index] :
         mtao::iterator::enumerate(edge_indices)) {
        size_t offset = constraint_matrix_offsets.at(local_index);
        auto e = E(edge_index);

        // a bit of error checking for unused edges:
        auto it = COB.find(edge_index);
        if (it == COB.end()) {
            continue;
        }
        auto [neg_cell, pos_cell] = it->second;

        double target_value = target_fluxes.at(edge_index);
        auto samples = boundary_sample_indices_vec(edge_index);
        auto B = boundary_basis(edge_index);
        auto N = B.col(1);

        for (auto&& [cidx, sign] : {std::make_tuple(neg_cell, false),
                                    std::make_tuple(pos_cell, true)}) {
            if (cidx < 0) {
                continue;
            }
            // const auto& projector = cell_projectors.at(cidx);
            size_t cell_off = coefficient_size() * cidx;
            double orient = sign ? -1 : 1;
            for (auto&& [local_ind, sample_index] :
                 mtao::iterator::enumerate(samples)) {
                size_t coeff_col = coefficient_size() * (local_ind + offset);

                auto g = polynomial_grad_entries(cidx,
                                                 sample_position(sample_index));
                mtao::RowVecXd Nd = N.transpose() * g;

                double dx =
                    ((local_ind == 0 || local_ind == (samples.size() - 1))
                         ? .5
                         : 1.) /
                    (samples.size() - 1);
                for (int j = 0; j < Nd.size(); ++j) {
                    trips.emplace_back(local_index, cell_off + j,
                                       dx * orient * Nd(j));
                }
            }
        }
    }
    A.setFromTriplets(trips.begin(), trips.end());
    mtao::VecXd r(target_fluxes.size());
    for (auto&& [a, b] : mtao::iterator::enumerate(edge_indices)) {
        r(a) = target_fluxes.at(b);
    }
    Eigen::SparseMatrix<double> R = A * sample2cell_coefficients();

    return {R, r};
}
Eigen::SparseMatrix<double> PointwiseVEMMesh2::sample2cell_coefficients(
    const std::set<size_t>& disengaged_cells) const {
    Eigen::SparseMatrix<double> A(coefficient_size() * num_cells(),
                                  num_samples());
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(A.rows() * 8);  // randomly assume we have ~8 cells per row

    for (size_t cidx = 0; cidx < num_cells(); ++cidx) {
        auto indices = cell_sample_indices_vec(cidx);

        auto M = poly_projection_sample2poly(cidx, indices);
        size_t off = cidx * coefficient_size();
        // write the the projection matrix out
        for (int j = 0; j < M.rows(); ++j) {
            for (auto&& [k, sidx] : mtao::iterator::enumerate(indices)) {
                // for(int k = 0; k < M.cols(); ++k) {
                auto v = M(j, k);
                if (std::abs(v) > 1e-6) {
                    trips.emplace_back(off + j, sidx, v);
                }
            }
        }
    }
    A.setFromTriplets(trips.begin(), trips.end());
    spdlog::warn("S2C shape: {} {}", A.rows(), A.cols());
    return A;
}
