
#include "vem/three/fluidsim/fluidvem.hpp"

#include <mtao/eigen/stack.hpp>
#include <mtao/logging/stopwatch.hpp>
#include <vem/utils/cell_identifier.hpp>
#include <vem/utils/local_to_world_sparse_triplets.hpp>
#include <vem/utils/loop_over_active.hpp>
#include <vem/three/volumes.hpp>

#include "mtao/eigen/mat_to_triplets.hpp"
#include "mtao/eigen/sparse_block_diagonal_repmats.hpp"
#include "vem/three/fluidsim/cell.hpp"
#include "vem/polynomials/utils.hpp"
namespace {
bool is_sparse_finite(const Eigen::SparseMatrix<double> &A) {
    using II = typename Eigen::SparseMatrix<double>::InnerIterator;

    for (int o = 0; o < A.outerSize(); ++o) {
        for (II it(A, o); it; ++it) {
            if (!std::isfinite(it.value())) {
                return false;
            }
        }
    }
    return true;
}
}  // namespace

using namespace vem::polynomials::two;
namespace vem::three::fluidsim {

FluidVEM3::FluidVEM3(const VEMMesh3 &mesh, size_t velocity_max_degree)
    : _mesh(mesh),
      _velocity_indexer(mesh, velocity_max_degree),
      _pressure_indexer(mesh, velocity_max_degree + 1) {}

FluidVEM3Cell FluidVEM3::get_velocity_cell(size_t index) const {
    return FluidVEM3Cell{velocity_indexer(), index};
}
FluidVEM3Cell FluidVEM3::get_pressure_cell(size_t index) const {
    return FluidVEM3Cell{pressure_indexer(), index};
}
size_t FluidVEM3::pressure_sample_size() const {
    return _pressure_indexer.sample_size();
}
size_t FluidVEM3::pressure_flux_size() const {
    return _pressure_indexer.flux_size();
}
size_t FluidVEM3::pressure_moment_size() const {
    return _pressure_indexer.moment_size();
}
size_t FluidVEM3::pressure_monomial_size() const {
    return _pressure_indexer.monomial_size();
}

size_t FluidVEM3::velocity_sample_size() const {
    return 3 * velocity_stride_sample_size();
}
size_t FluidVEM3::velocity_stride_sample_size() const {
    return _velocity_indexer.sample_size();
}
size_t FluidVEM3::velocity_stride_flux_size() const {
    return _velocity_indexer.flux_size();
}
size_t FluidVEM3::velocity_stride_moment_size() const {
    return _velocity_indexer.moment_size();
}
size_t FluidVEM3::velocity_stride_monomial_size() const {
    return _velocity_indexer.monomial_size();
}
size_t FluidVEM3::cell_count() const { return mesh().cell_count(); }

const std::set<int> &FluidVEM3::active_cells() const { return _active_cells; }
void FluidVEM3::set_active_cells(std::set<int> c) {
    _active_cells = std::move(c);
}

bool FluidVEM3::is_active_cell(int index) const {
    if (_active_cells.empty()) {
        return index >= 0 && index < cell_count();
    } else {
        return _active_cells.contains(index);
    }
}

size_t FluidVEM3::active_cell_count() const {
    if (_active_cells.empty()) {
        return cell_count();
    } else {
        return _active_cells.size();
    }
}

// Func should be mtao::MatXd(const FluidVEM3Cell&) where the returned is
// poly x local_sample shaped
// template <typename Func>
// Eigen::SparseMatrix<double> FluidVEM3::pressure_sample_to_poly_cell_matrix(
//    Func &&f) const {
//    Eigen::SparseMatrix<double> A(pressure_monomial_size(),
//                                  pressure_sample_size());
//    std::vector<Eigen::Triplet<double>> trips;
//    // comprised of blocks of per-cell x per-col entries
//    double mean_row_fill = (double)(A.rows()) / cell_count();
//    double mean_col_fill = (double)(A.cols()) / cell_count();
//
//    double pseudo_mean_cell_density = mean_row_fill * mean_col_fill;
//
//    trips.reserve(int(pseudo_mean_cell_density * cell_count()));
//
//    utils::loop_over_active_indices(
//        _mesh.cell_count(), active_cells(), [&](size_t cell_index) {
//            auto c = get_pressure_cell(cell_index);
//            auto L = f(c);
//            auto RC = c.local_to_world_monomial_indices();
//            auto CC = c.local_to_world_sample_indices();
//            std::vector<int> A;
//            for (auto &&v : RC) {
//                A.emplace_back(v);
//            }
//            std::vector<int> B;
//
//            for (auto &&v : CC) {
//                B.emplace_back(v);
//            }
//            // spdlog::info("{} row indices: {}", A.size(), fmt::join(A,
//            ","));
//            // spdlog::info("{} col indices: {}", B.size(), fmt::join(B,
//            ","));
//            // spdlog::info("interior matrix shape: {}x{}", L.rows(),
//            L.cols());
//
//            utils::local_to_world_sparse_triplets(RC, CC, L, trips);
//        });
//
//    A.setFromTriplets(trips.begin(), trips.end());
//    return A;
//}
//
//
//// Func should be mtao::MatXd(const FluidVEM3Cell&) where the returned is
//// local_sample x local_sample shaped
// template <typename Func>
// Eigen::SparseMatrix<double> FluidVEM3::pressure_poly_to_poly_cell_matrix(
//    Func &&f) const {
//    Eigen::SparseMatrix<double> A(pressure_monomial_size(),
//                                  pressure_monomial_size());
//    std::vector<Eigen::Triplet<double>> trips;
//    // comprised of blocks of per-cell x per-col entries
//    double mean_row_fill = (double)(A.rows()) / cell_count();
//    double mean_col_fill = (double)(A.cols()) / cell_count();
//
//    double pseudo_mean_cell_density = mean_row_fill * mean_col_fill;
//
//    trips.reserve(int(pseudo_mean_cell_density * cell_count()));
//
//    utils::loop_over_active_indices(
//        _mesh.cell_count(), active_cells(), [&](size_t cell_index) {
//            auto c = get_pressure_cell(cell_index);
//            auto L = f(c);
//            auto RC = c.local_to_world_monomial_indices();
//            const auto &CC = RC;
//            utils::local_to_world_sparse_triplets(RC, CC, L, trips);
//        });
//
//    A.setFromTriplets(trips.begin(), trips.end());
//    return A;
//}
//
//// Func should be mtao::MatXd(const FluidVEM3Cell&) where the returned is
//// poly x local_sample shaped
// template <typename Func>
// Eigen::SparseMatrix<double>
// FluidVEM3::velocity_stride_sample_to_poly_cell_matrix(Func &&f) const {
//    Eigen::SparseMatrix<double> A(velocity_stride_monomial_size(),
//                                  velocity_stride_sample_size());
//    std::vector<Eigen::Triplet<double>> trips;
//    // comprised of blocks of per-cell x per-col entries
//    double mean_row_fill = (double)(A.rows()) / cell_count();
//    double mean_col_fill = (double)(A.cols()) / cell_count();
//
//    double pseudo_mean_cell_density = mean_row_fill * mean_col_fill;
//
//    trips.reserve(int(pseudo_mean_cell_density * cell_count()));
//
//    utils::loop_over_active_indices(
//        _mesh.cell_count(), active_cells(), [&](size_t cell_index) {
//            auto c = get_velocity_cell(cell_index);
//            auto L = f(c);
//            auto RC = c.local_to_world_monomial_indices();
//            auto CC = c.local_to_world_sample_indices();
//            utils::local_to_world_sparse_triplets(RC, CC, L, trips);
//        });
//
//    A.setFromTriplets(trips.begin(), trips.end());
//    return A;
//}

Eigen::SparseMatrix<double>
FluidVEM3::velocity_stride_to_pressure_monomial_map() const {
    Eigen::SparseMatrix<double> R(pressure_monomial_size(),
                                  velocity_stride_monomial_size());
    std::vector<Eigen::Triplet<double>> trips;
    std::vector<std::vector<Eigen::Triplet<double>>> per_cell_trips(
        mesh().cell_count());
    utils::loop_over_active_indices_tbb(
        mesh().cell_count(), active_cells(), [&](size_t cell_index) {
            auto pc = get_pressure_cell(cell_index);
            auto vc = get_velocity_cell(cell_index);
            auto prange = pc.monomial_indices();
            auto vrange = vc.monomial_indices();
            // auto p_l2w = pc.local_to_world_monomial_map();
            // auto v_l2w = vc.local_to_world_monomial_map();

            auto &pct = per_cell_trips[cell_index];
            pct.reserve(std::min(prange.size(), vrange.size()));
            for (auto &&[pi, vi] : mtao::iterator::zip(prange, vrange)) {
                pct.emplace_back(pi, vi, 1);
            }
        });
    size_t size = 0;
    for (auto &&pct : per_cell_trips) {
        size += pct.size();
    }
    trips.reserve(size);
    for (auto &&pct : per_cell_trips) {
        std::copy(pct.begin(), pct.end(), std::back_inserter(trips));
    }
    R.setFromTriplets(trips.begin(), trips.end());
    return R;
}

Eigen::SparseMatrix<double> FluidVEM3::sample_laplacian() const {
    auto sw =
        mtao::logging::hierarchical_stopwatch("FluidVEM3::sample_laplacian");
    return pressure_indexer().sample_laplacian(active_cells());
}

Eigen::SparseMatrix<double> FluidVEM3::sample_to_poly_l2() const {
    auto sw =
        mtao::logging::hierarchical_stopwatch("FluidVEM3::sample_to_poly_l2");
    return velocity_indexer().sample_to_poly_dirichlet(active_cells());
}

// pressure samples -> pressure poly
Eigen::SparseMatrix<double> FluidVEM3::sample_to_poly_dirichlet() const {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "FluidVEM3::sample_to_poly_dirichlet");
    return pressure_indexer().sample_to_poly_dirichlet(active_cells());
}

Eigen::SparseMatrix<double> FluidVEM3::poly_pressure_l2_grammian() const {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "FluidVEM3::poly_pressure_l2_grammian");
    return pressure_indexer().poly_l2_grammian(active_cells());
}

Eigen::SparseMatrix<double> FluidVEM3::poly_velocity_l2_grammian() const {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "FluidVEM3::poly_velocity_l2_grammian");
    return velocity_indexer().poly_l2_grammian(active_cells());
}
Eigen::SparseMatrix<double> FluidVEM3::sample_to_poly_codivergence() const {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "FluidVEM3::sample_to_poly_codivergence");
    auto G = pressure_indexer().monomial_indexer().gradient();

    auto poly_v2p = velocity_stride_to_pressure_monomial_map();
    auto l2Pis = sample_to_poly_l2();
    auto l2G = poly_pressure_l2_grammian();
    // spdlog::info("Mult of VS = l2g{}X{} v2p{}x{} pis{}x{}", l2G.rows(),
    //             l2G.cols(), poly_v2p.rows(), poly_v2p.cols(), l2Pis.rows(),
    //             l2Pis.cols());
    // v stuff
    Eigen::SparseMatrix<double> VS = l2G * poly_v2p * l2Pis;
    auto VSVS = mtao::eigen::sparse_block_diagonal_repmats(VS, 3);
    spdlog::info(
        "Sample2Poly COD Finite checks: grad {}, l2Pis {}, l2G {}, "
        "mono_map {}",
        is_sparse_finite(G), is_sparse_finite(l2Pis), is_sparse_finite(l2G),
        is_sparse_finite(poly_v2p));

    // spdlog::info("Mult of G{}x{} with VSVS{}x{}", G.rows(), G.cols(),
    //             VSVS.rows(), VSVS.cols());
    return G.transpose() * VSVS;
}
Eigen::SparseMatrix<double> FluidVEM3::sample_codivergence() const {
    auto sw =
        mtao::logging::hierarchical_stopwatch("FluidVEM3::sample_codivergence");
    Eigen::SparseMatrix<double> P = sample_to_poly_dirichlet();

    auto SPCod = sample_to_poly_codivergence();
    // spdlog::info("Mult of P{}x{} with SPCod{}x{}", P.rows(), P.cols(),
    //             SPCod.rows(), SPCod.cols());
    spdlog::info("Sample2Sample COD Finite checks: dPis {},s2pCod {}",
                 is_sparse_finite(P), is_sparse_finite(SPCod));
    return P.transpose() * SPCod;
}
Eigen::SparseMatrix<double> FluidVEM3::sample_gradient() const {
    auto E = polynomial_to_sample_evaluation_matrix(false);
    return mtao::eigen::sparse_block_diagonal_repmats(E, 3) *
           sample_to_poly_gradient();
}
Eigen::SparseMatrix<double> FluidVEM3::sample_to_poly_gradient() const {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "FluidVEM3::sample_to_poly_gradient");
    auto G = pressure_indexer().monomial_indexer().gradient();
    auto dPis = sample_to_poly_dirichlet();
    // auto gPis = sample_to_poly_l2();

    auto poly_v2p = velocity_stride_to_pressure_monomial_map();
    Eigen::SparseMatrix<double> B = poly_v2p.transpose();
    auto R =
        (mtao::eigen::sparse_block_diagonal_repmats(B, 3) * G * dPis).eval();
    /*
    spdlog::info(
        "Finite checks: grad {}, dPis {}, eval {}, s2pCod {}, mono_map {}, final
    sample grad {}", is_sparse_finite(G), is_sparse_finite(dPis),
    is_sparse_finite(E), is_sparse_finite(SPCod), is_sparse_finite(poly_v2p),
    is_sparse_finite(R));
    */
    return R;
}
Eigen::SparseMatrix<double> FluidVEM3::polynomial_to_sample_evaluation_matrix(
    bool use_pressure, CellWeightWeightMode mode) const {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "FluidVEM3::polynomial_to_sample_evaluation_matrix");
    mtao::VecXd weights(cell_count());
    switch (mode) {
        case CellWeightWeightMode::Unweighted:
            weights.setOnes();
            break;
        case CellWeightWeightMode::AreaWeighted:
            weights = volumes(_mesh);
    }

    std::vector<Eigen::Triplet<double>> trips;

    const FluxMomentIndexer3 &indexer =
        use_pressure ? pressure_indexer() : velocity_indexer();

    int row_count = indexer.sample_size();
    int col_count = indexer.monomial_size();

    // estimate each point hsa tocommunicate with the average monomial
    // count's number of monomials
    trips.reserve(row_count *
                  std::max<size_t>(1, double(col_count) / active_cell_count()));

    mtao::VecXd weight_sums = mtao::VecXd::Zero(row_count);

    utils::loop_over_active_indices(
        mesh().cell_count(), active_cells(), [&](size_t cell_index) {
            // do point samples
            FluidVEM3Cell c(indexer, cell_index);
            const double w = weights(cell_index);
            auto world_indices = c.local_to_world_sample_indices();
            const int monomial_offset = c.global_monomial_index_offset();
            auto D = c.monomial_evaluation();

            for (auto &&[loc, wor] : mtao::iterator::enumerate(world_indices)) {
                for (int mon_idx = 0; mon_idx < c.monomial_size(); ++mon_idx) {
                    trips.emplace_back(wor, monomial_offset + mon_idx,
                                       w * D(loc, mon_idx));
                }
                weight_sums(wor) += w;
            }
            if (c.moment_size() == 0) {
                return;
            }
            // return;

            // do moments
            // no weight stuff has to happen as each moment DOF is unique to
            // each cell
            const int moment_offset = c.global_moment_index_offset();

            auto integrals = indexer.monomial_indexer().monomial_integrals(
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
    Eigen::SparseMatrix<double> R(row_count, col_count);
    R.setFromTriplets(trips.begin(), trips.end());
    weight_sums = 1. / weight_sums.array();
    // add 0 weights for the cell DOFs that didn't require any weighting
    weight_sums = mtao::eigen::vstack(
        weight_sums,
        mtao::VecXd::Ones(indexer.sample_size() - weight_sums.rows()));
    // std::cout << "Weights:\n";
    // std::cout << weight_sums.transpose() << std::endl;
    // std::cout << "R:\n";
    // std::cout << R << std::endl;
    return weight_sums.asDiagonal() * R;
}

// pressure to pressure grammian
// Eigen::SparseMatrix<double> FluidVEM3::per_cell_poly_dirichlet_grammian()
//    const {
//    return {};
//}
mtao::ColVecs4d FluidVEM3::velocity_weighted_face_samples(
    int face_index) const {
    return velocity_indexer().weighted_face_samples(face_index, 50);
}

mtao::ColVecs4d FluidVEM3::pressure_weighted_face_samples(
    int face_index) const {
    return pressure_indexer().weighted_face_samples(face_index);
}
}  // namespace vem::fluidsim_3d
