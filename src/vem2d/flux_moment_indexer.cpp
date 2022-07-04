#include "vem/flux_moment_indexer.hpp"

#include <mtao/geometry/interpolation/radial_basis_function.hpp>
#include <mtao/quadrature/gauss_lobatto.hpp>
#include <vem/utils/local_to_world_sparse_triplets.hpp>
#include <vem/utils/loop_over_active.hpp>

#include "vem/utils/boundary_facets.hpp"
#include "vem/utils/coefficient_accumulator.hpp"
#include "vem/utils/dehomogenize_vector_points.hpp"

namespace {
std::vector<size_t> offset_values(const std::vector<size_t> &o, size_t offset) {
    std::vector<size_t> ret;
    ret.reserve(o.size());
    std::transform(o.begin(), o.end(), std::back_inserter(ret),
                   [offset](size_t val) -> size_t {
                       if (val >= -offset) {
                           return val + offset;
                       } else {
                           return -1;
                       }
                   });
    return ret;
}

std::vector<size_t> max_edge_samples(const vem::VEMMesh2 &mesh,
                                     const std::vector<size_t> &o) {
    std::vector<size_t> edge_degrees(mesh.edge_count(), 0);
    for (auto &&[degree, fbm] :
         mtao::iterator::zip(o, mesh.face_boundary_map)) {
        if (degree == size_t(-1)) {
            continue;
        }
        for (auto &&[face, sign] : fbm) {
            size_t &v = edge_degrees.at(face);
            v = std ::max(degree, v);
        }
    }
    for (auto &&[idx, deg] : mtao::iterator::enumerate(edge_degrees)) {
        if (deg == size_t(-1)) {
            spdlog::error("Edge {} had deg {}", idx, deg);
        }
    }
    return edge_degrees;
}
}  // namespace

namespace vem {
template <typename Func>
Eigen::SparseMatrix<double> FluxMomentIndexer::sample_to_poly_cell_matrix(
    Func &&f, const std::set<int> &active_cells) const {
    Eigen::SparseMatrix<double> A(monomial_size(), sample_size());
    std::vector<Eigen::Triplet<double>> trips;
    std::vector<std::vector<Eigen::Triplet<double>>> per_cell_trips(
        mesh().cell_count());

    utils::loop_over_active_indices_tbb(
        _mesh.cell_count(), active_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto L = f(c);
            auto RC = c.local_to_world_monomial_indices();
            auto CC = c.local_to_world_sample_indices();
            auto &pct = per_cell_trips[cell_index];
            utils::local_to_world_sparse_triplets(RC, CC, L, pct);
        });

    size_t size = 0;
    for (auto &&pct : per_cell_trips) {
        size += pct.size();
    }
    trips.reserve(size);
    for (auto &&pct : per_cell_trips) {
        std::copy(pct.begin(), pct.end(), std::back_inserter(trips));
    }
    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

bool FluxMomentIndexer::is_edge_inactive(int edge_index) const {
    return flux_indexer().degree(edge_index) == size_t(-1);
}

template <typename Func>
Eigen::SparseMatrix<double> FluxMomentIndexer::sample_to_sample_cell_matrix(
    Func &&f, const std::set<int> &active_cells) const {
    Eigen::SparseMatrix<double> A(sample_size(), sample_size());
    std::vector<Eigen::Triplet<double>> trips;

    std::vector<std::vector<Eigen::Triplet<double>>> per_cell_trips(
        mesh().cell_count());

    utils::loop_over_active_indices_tbb(
        _mesh.cell_count(), active_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto L = f(c);
            auto RC = c.local_to_world_sample_indices();
            const auto &CC = RC;
            auto &pct = per_cell_trips[cell_index];
            utils::local_to_world_sparse_triplets(RC, CC, L, pct);
        });

    size_t size = 0;
    for (auto &&pct : per_cell_trips) {
        size += pct.size();
    }
    trips.reserve(size);
    for (auto &&pct : per_cell_trips) {
        std::copy(pct.begin(), pct.end(), std::back_inserter(trips));
    }
    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

// Func should be mtao::MatXd(const FluxMomentIndexerCell&) where the returned
// is local_sample x local_sample shaped
template <typename Func>
Eigen::SparseMatrix<double> FluxMomentIndexer::poly_to_poly_cell_matrix(
    Func &&f, const std::set<int> &active_cells) const {
    Eigen::SparseMatrix<double> A(monomial_size(), monomial_size());
    std::vector<Eigen::Triplet<double>> trips;
    std::vector<std::vector<Eigen::Triplet<double>>> per_cell_trips(
        mesh().cell_count());
    utils::loop_over_active_indices_tbb(
        _mesh.cell_count(), active_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto L = f(c);
            auto RC = c.local_to_world_monomial_indices();
            const auto &CC = RC;
            auto &pct = per_cell_trips[cell_index];
            utils::local_to_world_sparse_triplets(RC, CC, L, pct);
        });

    size_t size = 0;
    for (auto &&pct : per_cell_trips) {
        size += pct.size();
    }
    trips.reserve(size);
    for (auto &&pct : per_cell_trips) {
        std::copy(pct.begin(), pct.end(), std::back_inserter(trips));
    }
    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

FluxMomentVEM2Cell FluxMomentIndexer::get_cell(size_t index) const {
    return FluxMomentVEM2Cell(*this, index);
}
FluxMomentIndexer::FluxMomentIndexer(const VEMMesh2 &mesh, size_t max_degree)
    : FluxMomentIndexer(mesh,
                        std::vector<size_t>(mesh.cell_count(), max_degree)) {}

FluxMomentIndexer::FluxMomentIndexer(const VEMMesh2 &mesh,
                                     std::vector<size_t> max_degrees)
    : _mesh(mesh),
      _flux_indexer(mesh,
                    offset_values(max_edge_samples(mesh, max_degrees), -1)),
      _moment_indexer(mesh, offset_values(max_degrees, -2)),
      _monomial_indexer(mesh, std::move(max_degrees)) {}

size_t FluxMomentIndexer::sample_size() const {
    return flux_size() + moment_size();
}
size_t FluxMomentIndexer::flux_size() const {
    return flux_indexer().num_coefficients();
}
size_t FluxMomentIndexer::moment_size() const {
    return moment_indexer().num_coefficients();
}
size_t FluxMomentIndexer::monomial_size() const {
    return monomial_indexer().num_coefficients();
}

mtao::ColVecs3d FluxMomentIndexer::weighted_edge_samples(int edge_index) const {
    return weighted_edge_samples(edge_index,
                                 2 * _flux_indexer.degree(edge_index) + 1);
}
mtao::ColVecs3d FluxMomentIndexer::weighted_edge_samples(int edge_index,
                                                         int max_degree) const {
    if (max_degree < 1) {
        return {};
    }
    auto &&[P, W] = mtao::quadrature::gauss_lobatto_data<double>(max_degree);

    auto e = _mesh.E.col(edge_index);

    mtao::ColVecs3d V(3, P.size());
    auto a = _mesh.V.col(e(0));
    auto b = _mesh.V.col(e(1));

    // V.row(2) = .5 * (a - b).norm() * mtao::eigen::stl2eigen(W).transpose();
    // set to 0-1 interval
    V.row(2) = .5 * mtao::eigen::stl2eigen(W).transpose();
    auto pret = mtao::eigen::stl2eigen(P);
    auto t = (.5 * pret.array() + .5).matrix();
    V.topRows<2>() =
        a * (1 - t.array()).matrix().transpose() + b * t.transpose();

    // hardcode the interior ones and lerp the interior
    return V;
}

template <int D>
mtao::ColVectors<double, D + 1>
FluxMomentIndexer::_homogeneous_coefficients_from_point_values(
    const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
    const std::function<double(const mtao::Vec2d &, const mtao::Vec2d &)> &rbf,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    utils::CoefficientAccumulator<FluxMomentIndexer> ca(*this);
    return ca.homogeneous_coefficients_from_point_values(V, P, cell_particles,
                                                         active_cells, rbf);
}
template <int D>
mtao::ColVectors<double, D + 1>
FluxMomentIndexer::_homogeneous_coefficients_from_point_sample_function(
    const std::function<typename mtao::Vector<double, D>(const mtao::Vec2d &)>
        &f,
    const mtao::ColVecs2d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    utils::CoefficientAccumulator<FluxMomentIndexer> ca(*this);
    return ca.homogeneous_coefficients_from_point_function(f, P, cell_particles,
                                                           active_cells);
}

template <int D>
mtao::ColVectors<double, D>
FluxMomentIndexer::_coefficients_from_point_sample_function(
    const std::function<typename mtao::Vector<double, D>(const mtao::Vec2d &)>
        &f,
    const mtao::ColVecs2d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    auto R = _homogeneous_coefficients_from_point_sample_function(
        f, P, cell_particles, active_cells);
    return utils::dehomogenize_vector_points(R);
}
template <int D>
mtao::ColVectors<double, D> FluxMomentIndexer::_coefficients_from_point_values(
    const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
    const std::function<double(const mtao::Vec2d &, const mtao::Vec2d &)> &rbf,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    auto R = _homogeneous_coefficients_from_point_values(
        V, P, rbf, cell_particles, active_cells);
    return utils::dehomogenize_vector_points(R);
}

mtao::VecXd FluxMomentIndexer::coefficients_from_point_sample_function(
    const std::function<double(const mtao::Vec2d &)> &f,
    const mtao::ColVecs2d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    return _coefficients_from_point_sample_function<1>(
               [&](const mtao::Vec2d &p) -> mtao::Vector<double, 1> {
                   return mtao::Vector<double, 1>(f(p));
               },
               P, cell_particles, active_cells)
        .transpose();
}

mtao::ColVecs2d FluxMomentIndexer::coefficients_from_point_sample_function(
    const std::function<mtao::Vec2d(const mtao::Vec2d &)> &f,
    const mtao::ColVecs2d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    return _coefficients_from_point_sample_function<2>(
        [&](const mtao::Vec2d &p) -> mtao::Vector<double, 2> { return f(p); },
        P, cell_particles, active_cells);
}
mtao::VecXd FluxMomentIndexer::coefficients_from_point_values(
    const mtao::VecXd &V, const mtao::ColVecs2d &P,
    const std::function<double(const mtao::Vec2d &, const mtao::Vec2d &)> &rbf,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    return _coefficients_from_point_values<1>(V, P, rbf, cell_particles,
                                              active_cells);
}

mtao::ColVecs2d FluxMomentIndexer::coefficients_from_point_values(
    const mtao::ColVecs2d &V, const mtao::ColVecs2d &P,
    const std::function<double(const mtao::Vec2d &, const mtao::Vec2d &)> &rbf,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    return _coefficients_from_point_values<2>(V, P, rbf, cell_particles,
                                              active_cells);
}

Eigen::SparseMatrix<double> FluxMomentIndexer::sample_laplacian(
    const std::set<int> &active_cells) const {
    return sample_to_sample_cell_matrix(
        [&](const FluxMomentVEM2Cell &cell) -> mtao::MatXd {
            auto Pis = cell.dirichlet_projector();
            auto G = cell.monomial_dirichlet_grammian();
            auto E = cell.dirichlet_projector_error();

            auto R = Pis.transpose() * G * Pis + E.transpose() * E;
            return R;
        },
        active_cells);
}

Eigen::SparseMatrix<double> FluxMomentIndexer::poly_laplacian(
    const std::set<int> &active_cells) const {
    return poly_to_poly_cell_matrix(
        [&](const FluxMomentVEM2Cell &cell) -> mtao::MatXd {
            return cell.monomial_dirichlet_grammian();
        },
        active_cells);
}

Eigen::SparseMatrix<double> FluxMomentIndexer::sample_to_poly_l2(
    const std::set<int> &active_cells) const {
    return sample_to_poly_cell_matrix(
        [&](const FluxMomentVEM2Cell &cell) -> mtao::MatXd {
            auto R = cell.l2_projector();
            return R;
        },
        active_cells);
}
Eigen::SparseMatrix<double> FluxMomentIndexer::sample_to_poly_dirichlet(
    const std::set<int> &active_cells) const {
    return sample_to_poly_cell_matrix(
        [&](const FluxMomentVEM2Cell &cell) -> mtao::MatXd {
            auto R = cell.dirichlet_projector();
            return R;
        },
        active_cells);
}
Eigen::SparseMatrix<double> FluxMomentIndexer::poly_l2_grammian(
    const std::set<int> &active_cells) const {
    return poly_to_poly_cell_matrix(
        [&](const FluxMomentVEM2Cell &cell) -> mtao::MatXd {
            auto R = cell.monomial_l2_grammian();
            return R;
        },
        active_cells);
}
}  // namespace vem

