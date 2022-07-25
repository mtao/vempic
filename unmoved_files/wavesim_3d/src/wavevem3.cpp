
#include "vem/wavesim_3d/wavevem3.hpp"

#include <mtao/eigen/stack.hpp>
#include <vem/utils/cell_identifier.hpp>
#include <vem/utils/local_to_world_sparse_triplets.hpp>
#include <vem/utils/loop_over_active.hpp>
#include <vem/utils/parent_maps.hpp>
#include <vem/utils/volumes.hpp>

#include "mtao/eigen/mat_to_triplets.hpp"
#include "mtao/eigen/sparse_block_diagonal_repmats.hpp"
#include "vem/polynomial_utils.hpp"
#include "vem/wavesim_3d/wavevem3_cell.hpp"

using namespace vem::polynomials::two;
namespace vem::wavesim_3d {

WaveVEM3::WaveVEM3(const VEMMesh3 &mesh, size_t max_degree)
    : _mesh(mesh), _indexer(mesh, max_degree) {}

WaveVEM3Cell WaveVEM3::get_cell(size_t index) const {
    return WaveVEM3Cell{indexer(), index};
}
size_t WaveVEM3::sample_size() const { return _indexer.sample_size(); }
size_t WaveVEM3::flux_size() const { return _indexer.flux_size(); }
size_t WaveVEM3::moment_size() const { return _indexer.moment_size(); }
size_t WaveVEM3::monomial_size() const { return _indexer.monomial_size(); }

size_t WaveVEM3::cell_count() const { return mesh().cell_count(); }

const std::set<int> &WaveVEM3::active_cells() const { return _active_cells; }
void WaveVEM3::set_active_cells(std::set<int> c) {
    _active_cells = std::move(c);
}

bool WaveVEM3::is_active_cell(int index) const {
    if (_active_cells.empty()) {
        return index >= 0 && index < cell_count();
    } else {
        return _active_cells.contains(index);
    }
}

size_t WaveVEM3::active_cell_count() const {
    if (_active_cells.empty()) {
        return cell_count();
    } else {
        return _active_cells.size();
    }
}

// Func should be mtao::MatXd(const WaveVEM3Cell&) where the returned is
// poly x local_sample shaped
template <typename Func>
Eigen::SparseMatrix<double> WaveVEM3::sample_to_poly_cell_matrix(
    Func &&f) const {
    Eigen::SparseMatrix<double> A(monomial_size(), sample_size());
    std::vector<Eigen::Triplet<double>> trips;
    // comprised of blocks of per-cell x per-col entries
    double mean_row_fill = (double)(A.rows()) / cell_count();
    double mean_col_fill = (double)(A.cols()) / cell_count();

    double pseudo_mean_cell_density = mean_row_fill * mean_col_fill;

    trips.reserve(int(pseudo_mean_cell_density * cell_count()));

    utils::loop_over_active_indices(
        _mesh.cell_count(), active_cells(), [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto L = f(c);
            auto RC = c.local_to_world_monomial_indices();
            auto CC = c.local_to_world_sample_indices();
            std::vector<int> A;
            for (auto &&v : RC) {
                A.emplace_back(v);
            }
            std::vector<int> B;

            for (auto &&v : CC) {
                B.emplace_back(v);
            }
            // spdlog::info("{} row indices: {}", A.size(), fmt::join(A, ","));
            // spdlog::info("{} col indices: {}", B.size(), fmt::join(B, ","));
            // spdlog::info("interior matrix shape: {}x{}", L.rows(), L.cols());

            utils::local_to_world_sparse_triplets(RC, CC, L, trips);
        });

    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

template <typename Func>
Eigen::SparseMatrix<double> WaveVEM3::sample_to_sample_cell_matrix(
    Func &&f) const {
    Eigen::SparseMatrix<double> A(sample_size(), sample_size());
    std::vector<Eigen::Triplet<double>> trips;
    // comprised of blocks of per-cell x per-col entries
    double mean_row_fill = (double)(A.rows()) / cell_count();
    double mean_col_fill = (double)(A.cols()) / cell_count();

    double pseudo_mean_cell_density = mean_row_fill * mean_col_fill;

    trips.reserve(int(pseudo_mean_cell_density * cell_count()));

    utils::loop_over_active_indices(
        _mesh.cell_count(), active_cells(), [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto L = f(c);
            auto RC = c.local_to_world_sample_indices();
            const auto &CC = RC;
            utils::local_to_world_sparse_triplets(RC, CC, L, trips);
        });

    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

// Func should be mtao::MatXd(const WaveVEM3Cell&) where the returned is
// local_sample x local_sample shaped
template <typename Func>
Eigen::SparseMatrix<double> WaveVEM3::poly_to_poly_cell_matrix(Func &&f) const {
    Eigen::SparseMatrix<double> A(monomial_size(), monomial_size());
    std::vector<Eigen::Triplet<double>> trips;
    // comprised of blocks of per-cell x per-col entries
    double mean_row_fill = (double)(A.rows()) / cell_count();
    double mean_col_fill = (double)(A.cols()) / cell_count();

    double pseudo_mean_cell_density = mean_row_fill * mean_col_fill;

    trips.reserve(int(pseudo_mean_cell_density * cell_count()));

    utils::loop_over_active_indices(
        _mesh.cell_count(), active_cells(), [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto L = f(c);
            auto RC = c.local_to_world_monomial_indices();
            const auto &CC = RC;
            utils::local_to_world_sparse_triplets(RC, CC, L, trips);
        });

    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

Eigen::SparseMatrix<double> WaveVEM3::sample_laplacian() const {
    return sample_to_sample_cell_matrix(
        [&](const WaveVEM3Cell &cell) -> mtao::MatXd {
            auto Pis = cell.dirichlet_projector();
            auto G = cell.monomial_dirichlet_grammian();
            auto E = cell.dirichlet_projector_error();

            auto R = Pis.transpose() * G * Pis + E.transpose() * E;
            return R;
        });
}

Eigen::SparseMatrix<double> WaveVEM3::sample_to_poly_l2() const {
    return sample_to_poly_cell_matrix(
        [&](const WaveVEM3Cell &cell) -> mtao::MatXd {
            auto R = cell.l2_projector();
            return R;
        });
}

// pressure samples -> pressure poly
Eigen::SparseMatrix<double> WaveVEM3::sample_to_poly_dirichlet() const {
    return sample_to_poly_cell_matrix(
        [&](const WaveVEM3Cell &cell) -> mtao::MatXd {
            auto R = cell.dirichlet_projector();
            return R;
        });
    // spdlog::info("{}x{} expected, got {}x{}", RC.size(), CC.size(),
    //             L.rows(), L.cols());
}

Eigen::SparseMatrix<double> WaveVEM3::polynomial_to_sample_evaluation_matrix(
    CellWeightWeightMode mode) const {
    mtao::VecXd weights(cell_count());
    switch (mode) {
        case CellWeightWeightMode::Unweighted:
            weights.setOnes();
            break;
        case CellWeightWeightMode::AreaWeighted:
            weights = utils::volumes(_mesh);
    }

    std::vector<Eigen::Triplet<double>> trips;

    int row_count = indexer().sample_size();
    int col_count = indexer().monomial_size();

    // estimate each point hsa tocommunicate with the average monomial
    // count's number of monomials
    trips.reserve(row_count *
                  std::max<size_t>(1, double(col_count) / active_cell_count()));

    mtao::VecXd weight_sums = mtao::VecXd::Zero(row_count);

    utils::loop_over_active_indices(
        mesh().cell_count(), active_cells(), [&](size_t cell_index) {
            // do point samples
            auto c = get_cell(cell_index);
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

            auto integrals = indexer().monomial_indexer().monomial_integrals(
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
        weight_sums, mtao::VecXd::Ones(.sample_size() - weight_sums.rows()));
    // std::cout << "Weights:\n";
    // std::cout << weight_sums.transpose() << std::endl;
    // std::cout << "R:\n";
    // std::cout << R << std::endl;
    return weight_sums.asDiagonal() * R;
}

}  // namespace vem::wavesim_3d
