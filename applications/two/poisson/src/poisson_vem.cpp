#include "vem/poisson_2d/poisson_vem.hpp"

#include <omp.h>

#include <mtao/eigen/mat_to_triplets.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
//#include <vem/cells/l2_cell.hpp>
#include <vem/edge_lengths.hpp>
#include <vem/monomial_cell_integrals.hpp>
#include <vem/monomial_edge_integrals.hpp>
#include <vem/normals.hpp>
#include <vem/polynomial_gradient.hpp>
#include <vem/utils/cell_identifier.hpp>
#include <vem/utils/local_to_world_sparse_triplets.hpp>
#include <vem/utils/loop_over_active.hpp>
#include <vem/utils/parent_maps.hpp>
#include <vem/utils/volumes.hpp>

namespace vem::poisson_2d {

PoissonVEM2::PoissonVEM2(const VEMMesh2 &_mesh, size_t max_degree)
    : PoissonVEM2(_mesh, max_degree, std::max<size_t>(0, max_degree - 1)) {}
PoissonVEM2::PoissonVEM2(const VEMMesh2 &_mesh, size_t max_degree,
                         size_t edge_subsamples)
    : PoissonVEM2(_mesh, std::vector<size_t>(_mesh.cell_count(), max_degree),
                  std::vector<size_t>(_mesh.edge_count(), edge_subsamples),
                  std::vector<size_t>(_mesh.cell_count(),
                                      std::max<size_t>(0, max_degree - 2))) {}

// NOTE that this constructor has to move the diameters at the very end because
// two subojects hold diameters type objects
PoissonVEM2::PoissonVEM2(const VEMMesh2 &mesh, std::vector<size_t> max_degrees,
                         std::vector<size_t> edge_subsamples,
                         std::vector<size_t> moment_degrees,
                         std::vector<double> diameters)
    : _mesh(mesh),
      _monomial_indexer(_mesh, std::move(max_degrees), diameters),
      _point_sample_indexer(_mesh, std::move(edge_subsamples)),
      _moment_indexer(_mesh, std::move(moment_degrees), std::move(diameters)) {}

PoissonVEM2 PoissonVEM2::relative_order_mesh(int relative_order) const {
    auto degrees = _monomial_indexer.cell_degrees();

    for (auto &&deg : degrees) {
        deg = std::max<size_t>(0, deg - relative_order);
    }

    auto edge_subsamples = _point_sample_indexer.partition_offsets();
    for (auto &&k : edge_subsamples) {
        k = std::max<size_t>(0, k - relative_order);
    }

    auto diameters = _monomial_indexer.cell_diameters();

    auto moment_degrees = _moment_indexer.cell_degrees();

    for (auto &&k : moment_degrees) {
        k = std::max<size_t>(0, k - relative_order);
    }

    return PoissonVEM2(_mesh, std::move(degrees), std::move(edge_subsamples),
                       std::move(moment_degrees), std::move(diameters));
}
using namespace vem::polynomials::two;
PoissonVEM2Cell PoissonVEM2::get_cell(size_t index) const {
    return PoissonVEM2Cell{_mesh, index, _point_sample_indexer,
                           _monomial_indexer, _moment_indexer};
}
Eigen::SparseMatrix<double> PoissonVEM2::sample_to_polynomial_projection_matrix(
    const std::set<int> &used_cells) const {
    Eigen::SparseMatrix<double> A(_monomial_indexer.num_coefficients(),
                                  system_size());
    std::vector<Eigen::Triplet<double>> trips;

    double mean_monomial_size = (double)(_monomial_indexer.num_coefficients()) /
                                _monomial_indexer.num_partitions();

    trips.reserve(
        int(mean_monomial_size * 2 * _point_sample_indexer.num_coefficients()));

    utils::loop_over_active_indices(
        _mesh.cell_count(), used_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto Pi = c.Pis();

            auto CC = c.local_to_world_sample_indices();
            auto RC = c.local_to_world_monomial_indices();
            utils::local_to_world_sparse_triplets(RC, CC, Pi, trips);
        });
    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}
Eigen::SparseMatrix<double> PoissonVEM2::mass_matrix(
    const std::set<int> &used_cells) const {
    Eigen::SparseMatrix<double> A(system_size(), system_size());

    std::vector<Eigen::Triplet<double>> trips;

    double mean_sample_size = (double)(system_size()) / cell_count();
    double pseudo_mean_cell_density = mean_sample_size * mean_sample_size;

    trips.reserve(int(pseudo_mean_cell_density * cell_count()));

    utils::loop_over_active_indices(
        _mesh.cell_count(), used_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            mtao::MatXd L = c.M() + c.PErr();
            auto CC = c.local_to_world_sample_indices();
            const auto &RC = CC;
            utils::local_to_world_sparse_triplets(RC, CC, L, trips);
        });

    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}
Eigen::SparseMatrix<double> PoissonVEM2::stiffness_matrix(
    const std::set<int> &used_cells) const {
    return laplacian(used_cells);
}
Eigen::SparseMatrix<double> PoissonVEM2::laplacian(
    const std::set<int> &used_cells) const {
    return point_laplacian(used_cells);
}
Eigen::SparseMatrix<double> PoissonVEM2::point_laplacian(
    const std::set<int> &used_cells) const {
    Eigen::SparseMatrix<double> A(system_size(), system_size());
    std::vector<Eigen::Triplet<double>> trips;
    double mean_sample_size = (double)(system_size()) / cell_count();
    double pseudo_mean_cell_density = mean_sample_size * mean_sample_size;

    trips.reserve(int(pseudo_mean_cell_density * cell_count()));

    utils::loop_over_active_indices(
        _mesh.cell_count(), used_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto L = c.KEH();
            auto CC = c.local_to_world_sample_indices();
            const auto &RC = CC;
            utils::local_to_world_sparse_triplets(RC, CC, L, trips);
        });

    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

Eigen::SparseMatrix<double> PoissonVEM2::projection_error(
    const std::set<int> &used_cells) const {
    Eigen::SparseMatrix<double> A(system_size(), system_size());
    std::vector<Eigen::Triplet<double>> trips;
    double mean_sample_size = (double)(system_size()) / cell_count();
    double pseudo_mean_cell_density = mean_sample_size * mean_sample_size;

    trips.reserve(int(pseudo_mean_cell_density * cell_count()));

    utils::loop_over_active_indices(
        _mesh.cell_count(), used_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto L = c.PErr();
            auto CC = c.local_to_world_sample_indices();
            const auto &RC = CC;
            utils::local_to_world_sparse_triplets(RC, CC, L, trips);
        });
    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

Eigen::SparseMatrix<double> PoissonVEM2::poly_laplacian(
    const std::set<int> &used_cells) const {
    auto G = sample_to_poly_divergence(used_cells);
    auto M = mass_matrix(used_cells);
    auto MM = mtao::eigen::sparse_block_diagonal_repmats(M, 2);

    return G * MM * G.transpose();
    // return G * G.transpose();
}
// Eigen::SparseMatrix<double> PoissonVEM2::stiffness_matrix_sqrt(
//    const std::set<int>& used_cells) const {
//    Eigen::SparseMatrix<double> A(system_size(), system_size());
//
//    auto run = [&](int cell_index) {
//        auto c = get_cell(cell_index);
//        auto P = c.local_to_world_sample_map();
//        auto L = c.KEH_sqrt();
//        // std::cout << L << std::endl << std::endl;
//        Eigen::SparseMatrix<double> LP = (P * L).sparseView();
//        LP = LP * P.transpose();
//        A = A + LP;
//    };
//    if (used_cells.empty()) {
//        for (size_t cell_index = 0; cell_index < _mesh.cell_count();
//             ++cell_index) {
//            run(cell_index);
//        }
//    } else {
//        for (auto&& c : used_cells) {
//            run(c);
//        }
//    }
//    return A;
//}

int PoissonVEM2::system_size() const { return point_size() + moment_size(); }
int PoissonVEM2::cell_count() const { return _mesh.cell_count(); }
int PoissonVEM2::point_size() const {
    return _point_sample_indexer.num_coefficients();
}
int PoissonVEM2::moment_size() const {
    return _moment_indexer.num_coefficients();
}
int PoissonVEM2::monomial_size() const {
    return _monomial_indexer.num_coefficients();
}

std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> PoissonVEM2::kkt_system(
    const ScalarConstraints &constraints,
    const std::set<int> &used_cells) const {
    auto [C, Crhs] = polynomial_constraint_matrix(constraints, used_cells);
    std::cout << "Pointwise constraints:\n"
              << C << "\n"
              << "Constraint RHS\n"
              << Crhs << std::endl;
    // auto P = polynomial_to_sample_evaluation_matrix(
    //    CellWeightWeightMode::AreaWeighted, used_cells);
    // std::cout << "Poly eval mat:\n" << P << std::endl;
    // C = (C * P).eval();
    std::cout << "Poly-wise constraint matrix:\n" << C << std::endl;
    auto K = poly_laplacian(used_cells);

    int sys_size = K.rows();

    std::vector<Eigen::Triplet<double>> trips = mtao::eigen::mat_to_triplets(K);
    trips.reserve(trips.size() + 2 * C.nonZeros());

    for (int k = 0; k < C.outerSize(); ++k) {
        for (decltype(C)::InnerIterator it(C, k); it; ++it) {
            trips.emplace_back(sys_size + it.row(), it.col(), it.value());
            trips.emplace_back(it.col(), sys_size + it.row(), it.value());
        }
    }

    size_t size = sys_size + C.rows();
    Eigen::SparseMatrix<double> R(size, size);
    R.setFromTriplets(trips.begin(), trips.end());

    mtao::VecXd rhs(sys_size);
    rhs.setZero();
    return {R, mtao::eigen::vstack(rhs, Crhs)};
}
std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd>
PoissonVEM2::point_constraint_matrix(const ScalarConstraints &constraints,
                                     const std::set<int> &used_cells) const {
    auto vertex_face_map = vem::utils::vertex_faces(_mesh);
    auto edge_face_map = vem::utils::edge_faces(_mesh);
    std::vector<Eigen::Triplet<double>> trips;

    std::list<double> rhs_values;
    auto cur_constraint_pos = [&]() -> size_t { return rhs_values.size(); };
    auto add_rhs_value = [&](double value) { rhs_values.emplace_back(value); };
    for (auto &&[vidx, value] : constraints.pointwise_dirichlet) {
        trips.emplace_back(cur_constraint_pos(), vidx, 1);
        add_rhs_value(value);
    }
    if (constraints.mean_value) {
    }

    auto N = normals(_mesh);
    for (auto &&[edge_idx, value] : constraints.edge_integrated_flux_neumann) {
        auto inds = _point_sample_indexer.ordered_edge_indices(edge_idx);
        auto W =
            mtao::quadrature::gauss_lobatto_sample_weights<double>(inds.size());
        auto n = N.col(edge_idx);
        auto e = _mesh.E.col(edge_idx);
        auto a = _mesh.V.col(e(0));
        auto b = _mesh.V.col(e(1));
        // factor because gauss lobatto is defined on [-1,1]
        double weight_scale = .5 * (b - a).norm();
        for (auto &&[fidx, sign] : edge_face_map.at(edge_idx)) {
            if (!used_cells.empty() && !used_cells.contains(fidx)) {
                continue;
            }
            auto c = get_cell(fidx);
            auto reindexer = c.world_to_local_point_indices();
            auto Pi = c.Pis();

            int num_monomials = c.monomial_size();
            auto G = polynomials::two::gradient(c.monomial_degree());
            G /= c.diameter();
            // std::cout << "Grad\n" << G << std::endl;
            // monomial func -> n \cdot grad in monomials
            Eigen::SparseMatrix<double> GN =
                (sign ? -1 : 1) * (n(0) * G.topRows(num_monomials) +
                                   n(1) * G.bottomRows(num_monomials));

            // std::cout << "Grad normal:\n " << GN << std::endl;

            auto D = c.D();
            // spdlog::info("Mat sizes: {} {}x{} {}x{} {}x{}", W.size(),
            // D.rows(),D.cols(),GN.rows(),GN.cols(),Pi.rows(),Pi.cols());
            // mtao::RowVecXd v = weight_scale *
            //                   mtao::eigen::stl2eigen(W).transpose() * D *
            //                   GN * Pi;

            auto point_indices = c.point_indices();
            for (auto &&[weight, point_ind] : mtao::iterator::zip(W, inds)) {
                int local_ind = reindexer.at(point_ind);
                mtao::RowVecXd v =
                    weight * weight_scale * D.row(local_ind) * GN * Pi;
                for (auto &&[local, global] :
                     mtao::iterator::enumerate(point_indices)) {
                    trips.emplace_back(cur_constraint_pos(), global, v(local));
                }

                trips.emplace_back();
            }
            // for (auto&& [ind, w] : mtao::iterator::zip(inds, v)) {
            //    trips.emplace_back(cur_constraint_pos(), ind, w);
            //}

            add_rhs_value(value);
        }
    }

    Eigen::SparseMatrix<double> R(cur_constraint_pos(), system_size());
    R.setFromTriplets(trips.begin(), trips.end());
    mtao::VecXd Rv(cur_constraint_pos());
    std::copy(rhs_values.begin(), rhs_values.end(), Rv.data());
    // std::cout << R << std::endl;
    return {R, Rv};
}
std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd>
PoissonVEM2::polynomial_constraint_matrix(
    const ScalarConstraints &constraints,
    const std::set<int> &used_cells) const {
    auto vertex_face_map = vem::utils::vertex_faces(_mesh);
    auto edge_face_map = vem::utils::edge_faces(_mesh);
    std::vector<Eigen::Triplet<double>> trips;

    std::list<double> rhs_values;
    auto cur_constraint_pos = [&]() -> size_t { return rhs_values.size(); };
    auto add_rhs_value = [&](double value) { rhs_values.emplace_back(value); };

    auto N = normals(_mesh);
    for (auto &&[edge_idx, value] : constraints.edge_integrated_flux_neumann) {
        auto inds = _point_sample_indexer.ordered_edge_indices(edge_idx);
        auto W =
            mtao::quadrature::gauss_lobatto_sample_weights<double>(inds.size());
        auto n = N.col(edge_idx);
        auto e = _mesh.E.col(edge_idx);
        auto a = _mesh.V.col(e(0));
        auto b = _mesh.V.col(e(1));
        // factor because gauss lobatto is defined on [-1,1]
        double weight_scale = .5 * (b - a).norm();
        for (auto &&[fidx, sign] : edge_face_map.at(edge_idx)) {
            if (!used_cells.empty() && !used_cells.contains(fidx)) {
                continue;
            }
            auto c = get_cell(fidx);
            auto reindexer = c.world_to_local_point_indices();
            auto Pi = c.Pis();

            int num_monomials = c.monomial_size();
            auto G = polynomials::two::gradient(c.monomial_degree());
            G /= c.diameter();
            // std::cout << "Grad\n" << G << std::endl;
            // monomial func -> n \cdot grad in monomials
            Eigen::SparseMatrix<double> GN =
                (sign ? -1 : 1) * (n(0) * G.topRows(num_monomials) +
                                   n(1) * G.bottomRows(num_monomials));

            auto edge_integrals =
                vem::single_edge_scaled_monomial_edge_integrals(
                    _mesh, fidx, edge_idx, c.diameter(), c.monomial_degree());
            mtao::RowVecXd row =
                mtao::eigen::stl2eigen(edge_integrals).transpose() * GN;
            auto [start, end] = monomial_indexer().coefficient_range(fidx);
            for (int j = 0; j < row.size(); ++j) {
                trips.emplace_back(cur_constraint_pos(), start + j, row(j));
            }

            // std::cout << "Grad normal:\n " << GN << std::endl;

            auto D = c.D();

            add_rhs_value(value);
        }
    }

    Eigen::SparseMatrix<double> R(cur_constraint_pos(), monomial_size());
    R.setFromTriplets(trips.begin(), trips.end());
    mtao::VecXd Rv(cur_constraint_pos());
    std::copy(rhs_values.begin(), rhs_values.end(), Rv.data());
    // std::cout << R << std::endl;
    return {R, Rv};
}
Eigen::SparseMatrix<double> PoissonVEM2::polynomial_to_sample_evaluation_matrix(
    CellWeightWeightMode mode, const std::set<int> &used_cells) const {
    mtao::VecXd weights;
    switch (mode) {
        case CellWeightWeightMode::Unweighted:
            weights.setConstant(1);
            break;
        case CellWeightWeightMode::AreaWeighted:
            weights = utils::volumes(_mesh);
    }

    std::vector<Eigen::Triplet<double>> trips;
    // estimate each point hsa tocommunicate with the average monomial
    // count's number of monomials
    trips.reserve(_mesh.cell_count() *
                  std::max<size_t>(1, _monomial_indexer.num_coefficients() /
                                          _mesh.cell_count()));
    mtao::VecXd weight_sums = mtao::VecXd::Zero(system_size());
    const int global_moment_offset = point_size();
    utils::loop_over_active_indices(
        _mesh.cell_count(), used_cells, [&](size_t cell_index) {
            // do point samples
            auto c = get_cell(cell_index);
            const double w = weights(cell_index);
            auto world_indices = c.local_to_world_sample_indices();
            const int monomial_offset = c.global_monomial_index_offset();
            auto D = c.D();
            for (auto &&[loc, wor] : mtao::iterator::enumerate(world_indices)) {
                for (int mon_idx = 0; mon_idx < c.monomial_size(); ++mon_idx) {
                    trips.emplace_back(wor, monomial_offset + mon_idx,
                                       w * D(loc, mon_idx));
                }
                weight_sums(wor) += w;
            }

            // do moments
            // no weight stuff has to happen as each moment DOF is unique to
            // each cell
            const int moment_offset =
                global_moment_offset +
                _moment_indexer.coefficient_offset(cell_index);

            auto integrals = _monomial_indexer.monomial_integrals(
                cell_index, c.monomial_degree() + c.moment_degree());
            // spdlog::info("Integrals: {} from mon and mom degrees {} {}",
            //        fmt::join(integrals, ","), c.monomial_degree(),
            //             c.moment_degree());
            for (int j = 0; j < c.moment_size(); ++j) {
                int row = moment_offset + j;
                auto [mxexp, myexp] = index_to_exponents(j);
                for (int k = 0; k < c.monomial_size(); ++k) {
                    int col = k + monomial_offset;
                    auto [Mxexp, Myexp] = index_to_exponents(k);

                    const int xexp = mxexp + Mxexp;
                    const int yexp = myexp + Myexp;
                    int integral_index = exponents_to_index(xexp, yexp);
                    // spdlog::info(
                    //    "Integral index {} (({}){} {} + ({}){} {} = {} {}) /
                    //    {}", integral_index, j, mxexp, myexp, k, Mxexp, Myexp,
                    //    xexp, yexp, integrals.size());
                    trips.emplace_back(row, col, integrals(integral_index));
                }
                // weight_sums(row) += w;
            }
        });
    /*
    auto vertex_face_map = vem::utils::vertex_faces(_mesh);
    for (auto&& [vertex_index, faces] : vertex_face_map) {
        mtao::Vec2d p =
    _point_sample_indexer.get_position(vertex_index); double vweight_sum
    = 0; for (auto&& face : faces) { vweight_sum += weights(face);
        }
        if (vweight_sum == 0) {
            vweight_sum = 1;
        }
        for (auto&& face : faces) {
            auto c = get_cell(face_index);
            int offset =
    _monomial_indexer.coefficient_offset(face_index);

            mtao::VecXd mono_weighted_vals =
                c.evaluate_monomials(p) * weights(face) / vweight_sum;

            for (int j = 0; j < mono_weighted_vals.size(); ++j) {
                trips.emplace_back(vertex_index, offset + j,
                                   mono_weighted_vals(j));
            }
        }
    }
    */
    Eigen::SparseMatrix<double> R(system_size(),
                                  _monomial_indexer.num_coefficients());
    R.setFromTriplets(trips.begin(), trips.end());
    weight_sums = 1. / weight_sums.array();
    // add 0 weights for the cell DOFs that didn't require any weighting
    weight_sums = mtao::eigen::vstack(
        weight_sums, mtao::VecXd::Ones(system_size() - weight_sums.rows()));
    // std::cout << "Weights:\n";
    // std::cout << weight_sums.transpose() << std::endl;
    // std::cout << "R:\n";
    // std::cout << R << std::endl;
    return weight_sums.asDiagonal() * R;
}
Eigen::SparseMatrix<double> PoissonVEM2::sample_to_poly_gradient(
    const std::set<int> &used_cells) const {
    // TODO: Add something for moments
    return _monomial_indexer.gradient() *
           sample_to_polynomial_projection_matrix(used_cells);
}
Eigen::SparseMatrix<double> PoissonVEM2::sample_to_sample_gradient(
    const std::set<int> &used_cells) const {
    auto P = polynomial_to_sample_evaluation_matrix(
        CellWeightWeightMode::AreaWeighted, used_cells);

    auto B = mtao::eigen::sparse_block_diagonal_repmats(P, 2);
    auto C = sample_to_poly_gradient(used_cells);
    return B * C;
}

Eigen::SparseMatrix<double> PoissonVEM2::sample_to_poly_divergence(
    const std::set<int> &used_cells) const {
    auto P = sample_to_polynomial_projection_matrix(used_cells);
    auto B = mtao::eigen::sparse_block_diagonal_repmats(P, 2);
    return _monomial_indexer.divergence() * B;
}
Eigen::SparseMatrix<double> PoissonVEM2::sample_to_sample_divergence(
    const std::set<int> &used_cells) const {
    auto P = polynomial_to_sample_evaluation_matrix(
        CellWeightWeightMode::AreaWeighted, used_cells);

    auto C = sample_to_poly_divergence();
    return P * C;
}
Eigen::SparseMatrix<double> PoissonVEM2::poly_to_sample_gradient(
    const std::set<int> &used_cells) const {
    auto P = polynomial_to_sample_evaluation_matrix(
        CellWeightWeightMode::AreaWeighted, used_cells);

    auto B = mtao::eigen::sparse_block_diagonal_repmats(P, 2);
    return B * _monomial_indexer.gradient();
}

Eigen::SparseMatrix<double> PoissonVEM2::poly_to_sample_lap_cogradient(
    const std::set<int> &used_cells) const {
    int mon_size = _monomial_indexer.num_coefficients();
    Eigen::SparseMatrix<double> A(system_size(), 2 * mon_size);
    int offset = 0;
    std::vector<Eigen::Triplet<double>> trips;
    utils::loop_over_active_indices(
        _mesh.cell_count(), used_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto P = c.local_to_world_sample_map();
            auto PM = c.local_to_world_monomial_map();
            auto G = c.CoGrad_mIn();
            int local_mon_size = c.monomial_size();
            int local_sys_size = c.local_system_size();
            auto Gx = G.leftCols(local_mon_size);
            auto Gy = G.rightCols(local_mon_size);
            for (int k = 0; k < P.outerSize(); ++k) {
                for (decltype(P)::InnerIterator pit(P, k); pit; ++pit) {
                    for (int k2 = 0; k2 < PM.outerSize(); ++k2) {
                        for (decltype(PM)::InnerIterator mit(PM, k2); mit;
                             ++mit) {
                            // A.transpose() * G * B
                            trips.emplace_back(pit.row(), mit.row(),
                                               Gx(pit.col(), mit.col()));
                            trips.emplace_back(pit.row(), mit.row() + mon_size,
                                               Gy(pit.col(), mit.col()));
                        }
                    }
                }
            }
        });
    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}
Eigen::SparseMatrix<double> PoissonVEM2::sample_to_sample_lap_cogradient(
    const std::set<int> &used_cells) const {
    /*
    auto G = poly_to_sample_lap_cogradient(used_cells);
    auto P = sample_to_polynomial_projection_matrix(used_cells);
    auto B = mtao::eigen::sparse_block_diagonal_repmats(P, 2);
    return G * B;
    */
    Eigen::SparseMatrix<double> A(system_size(), 2 * system_size());
    int offset = 0;
    std::vector<Eigen::Triplet<double>> trips;
    utils::loop_over_active_indices(
        _mesh.cell_count(), used_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto P = c.local_to_world_sample_map();
            auto G = c.CoGrad();
            int local_sys_size = c.local_system_size();
            auto Gx = G.leftCols(local_sys_size);
            auto Gy = G.rightCols(local_sys_size);
            // std::cout << "M dims: " << G.rows() << " " << G.cols() <<
            // std::endl;
            for (int k = 0; k < P.outerSize(); ++k) {
                for (decltype(P)::InnerIterator pit(P, k); pit; ++pit) {
                    for (int k2 = 0; k2 < P.outerSize(); ++k2) {
                        for (decltype(P)::InnerIterator mit(P, k2); mit;
                             ++mit) {
                            // A.transpose() * G * B
                            trips.emplace_back(pit.row(), mit.row(),
                                               Gx(pit.col(), mit.col()));
                            trips.emplace_back(pit.row(),
                                               mit.row() + system_size(),
                                               Gy(pit.col(), mit.col()));
                            // spdlog::info(
                            //    "G({} {}:{}) = M({} {}:{}) = {}:{}",
                            //    pit.row(), mit.row(), mit.row() +
                            //    system_size(), pit.col(), mit.col(), mit.col()
                            //    + local_sys_size, Gx(pit.col(), mit.col()),
                            //    Gy(pit.col(), mit.col()));
                        }
                    }
                }
            }
        });
    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}
Eigen::SparseMatrix<double> PoissonVEM2::sample_to_poly_lap_gradient(
    const std::set<int> &used_cells) const {
    int mon_size = _monomial_indexer.num_coefficients();
    Eigen::SparseMatrix<double> A(2 * mon_size, system_size());
    int offset = 0;
    std::vector<Eigen::Triplet<double>> trips;
    utils::loop_over_active_indices(
        _mesh.cell_count(), used_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto P = c.local_to_world_sample_map();
            auto PM = c.local_to_world_monomial_map();
            auto G = c.Grad_mOut();
            int local_mon_size = c.monomial_size();
            auto Gx = G.leftCols(local_mon_size);
            auto Gy = G.rightCols(local_mon_size);
            for (int k = 0; k < P.outerSize(); ++k) {
                for (decltype(P)::InnerIterator pit(P, k); pit; ++pit) {
                    for (int k2 = 0; k2 < PM.outerSize(); ++k2) {
                        for (decltype(PM)::InnerIterator mit(PM, k2); mit;
                             ++mit) {
                            // A.transpose() * G * B
                            trips.emplace_back(mit.row(), pit.row(),
                                               Gx(mit.col(), pit.col()));
                            trips.emplace_back(mit.row() + mon_size, pit.row(),
                                               Gy(mit.col(), pit.col()));
                        }
                    }
                }
            }
        });
    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}
Eigen::SparseMatrix<double> PoissonVEM2::sample_to_sample_lap_gradient(
    const std::set<int> &used_cells) const {
    auto G = sample_to_poly_lap_gradient(used_cells);
    auto P = polynomial_to_sample_evaluation_matrix(
        CellWeightWeightMode::AreaWeighted, used_cells);

    auto B = mtao::eigen::sparse_block_diagonal_repmats(P, 2);
    return B * G;
}

mtao::VecXd PoissonVEM2::coefficients_from_point_sample_function(
    const std::function<double(const mtao::Vec2d &)> &f) const {
    double val = (double)(_monomial_indexer.num_coefficients()) /
                     _monomial_indexer.num_partitions() +
                 2;
    return coefficients_from_point_sample_function(f, val * val);
}
mtao::VecXd PoissonVEM2::coefficients_from_point_sample_function(
    const std::function<double(const mtao::Vec2d &)> &f,
    int samples_per_cell) const {
    std::vector<std::set<int>> ownerships(cell_count());
    mtao::ColVecs2d points(2, samples_per_cell * cell_count());

    for (auto &&[idx, own] : mtao::iterator::enumerate(ownerships)) {
        auto c = get_cell(idx);
        auto bb = c.bounding_box();
        int offset = idx * samples_per_cell;
        for (int j = 0; j < samples_per_cell; ++j) {
            own.emplace(j + offset);
            auto p = points.col(j + offset) = bb.sample();
            while (!c.is_inside(p)) {
                p = bb.sample();
            }
        }
    }
    return coefficients_from_point_sample_function(f, points, ownerships);
}
mtao::VecXd PoissonVEM2::coefficients_from_point_sample_function(
    const std::function<double(const mtao::Vec2d &)> &f,
    const mtao::ColVecs2d &P,
    const std::vector<std::set<int>> &cell_particles) const {
    if (cell_particles.size() == 0) {
        auto new_cell_particles =
            utils::CellIdentifier<VEMMesh2>{mesh()}.cell_ownerships(P);
        if (!new_cell_particles.empty()) {
            return coefficients_from_point_sample_function(f, P,
                                                           new_cell_particles);
        } else {
            return {};
        }
    }

    mtao::VecXd A(system_size());
    for (int j = 0; j < point_size(); ++j) {
        mtao::Vec2d p = point_sample_indexer().get_position(j);
        A(j) = f(p);
    }

    int cell_index = 0;
#pragma omp parallel for
    for (cell_index = 0; cell_index < cell_particles.size(); ++cell_index) {
        const auto &particles = cell_particles[cell_index];
        // for (auto&& [cell_index, particles] :
        //     mtao::iterator::enumerate(cell_particles)) {
        auto c = get_cell(cell_index);
        // auto l2c = cells::L2Cell(c);

        if (c.moment_size() == 0) {
            continue;
        }

        const auto &momi = moment_indexer();

        auto pblock = A.segment(c.global_moment_index_offset() +
                                    momi.coefficient_offset(cell_index),
                                c.moment_size());

        pblock.setZero();
        auto samples = c.vertices();

        mtao::ColVecs2d LP(2, samples.size() + particles.size());
        mtao::VecXd V(samples.size() + particles.size());

        int index = 0;
        auto run = [&](auto &&pt) {
            double v = f(pt);
            pblock += c.evaluate_monomials_by_size(pblock.size(), pt) * v;

            LP.col(index) = pt;
            V(index) = v;
            index++;
        };

        for (auto &&p : particles) {
            run(P.col(p));
        }
        for (auto &&p : samples) {
            mtao::Vec2d pt = point_sample_indexer().get_position(p);
            run(pt);
        }

        // auto LQFit =
        //    l2c.unweighted_least_squares_coefficients(max_degree, LP, V);

        pblock /= particles.size() + samples.size();
        // bpblock = (c.monomial_l2_grammian() * LQFit).head(c.moment_size()) /
        //          c.volume();
    }
    return A;
}

mtao::VecXd PoissonVEM2::active_cell_polynomial_mask(
    const std::set<int> &used_cells) const {
    return monomial_indexer().partition_mask(used_cells);
}
}  // namespace vem::poisson_2d
