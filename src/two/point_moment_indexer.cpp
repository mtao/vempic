#include "vem/two/point_moment_indexer.hpp"
#include "vem/utils/coefficient_accumulator.hpp"
#include "vem/utils/dehomogenize_vector_points.hpp"

#include <vem/utils/local_to_world_sparse_triplets.hpp>
#include <vem/utils/loop_over_active.hpp>

#include "vem/two/boundary_facets.hpp"

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

std::vector<size_t> max_edge_samples(const vem::two::VEMMesh2 &mesh,
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
    return edge_degrees;
}
}  // namespace

namespace vem::utils {
template <>
template <int D>
mtao::ColVectors<double, D + 1> CoefficientAccumulator<two::PointMomentIndexer>::
    homogeneous_boundary_coefficients_from_point_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const {
    mtao::ColVectors<double, D + 1> R(D + 1, indexer.boundary_size());
    R.setZero();
    auto Dat = R.template topRows<D>();
    auto W = R.row(D);
    W.setConstant(1);
    tbb::parallel_for(size_t(0), indexer.point_sample_size(), [&](size_t j) {
        mtao::Vec2d p = indexer.point_sample_indexer().get_position(j);
        Dat.col(j) = f(p);
    });
    return R;
}

template <>
template <int D>
mtao::ColVectors<double, D + 1> CoefficientAccumulator<two::PointMomentIndexer>::
    homogeneous_boundary_coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells, const RBFFunc &rbf) const {
    mtao::ColVectors<double, D + 1> R(D + 1, indexer.boundary_size());

    auto Dat = R.template topRows<D>();
    auto W = R.row(D);

    W.setZero();
    auto B = Dat.leftCols(W.size());
    B.setZero();
    for (auto &&[cell_index, particles] :
         mtao::iterator::enumerate(cell_particles)) {
        auto c = indexer.get_cell(cell_index);
        for (auto &&point_sample_index : c.point_sample_indices()) {
            auto s =
                indexer.point_sample_indexer().get_position(point_sample_index);
            auto sample_vel = Dat.col(point_sample_index);
            double &weight_sum = W(point_sample_index);
            for (auto &&p : particles) {
                double weight =
                    rbf(s, P.col(p));  // * (p - center).normalized();

                weight_sum += weight;
                sample_vel += weight * V.col(p);
            }
        }
    }
    return R;
}
}
namespace vem::two {

PointMomentVEM2Cell PointMomentIndexer::get_cell(size_t index) const {
    return PointMomentVEM2Cell(*this, index);
}
// Func should be mtao::MatXd(const PointMomentIndexerCell&) where the returned
// is poly x local_sample shaped
template <typename Func>
Eigen::SparseMatrix<double> PointMomentIndexer::sample_to_poly_cell_matrix(
    Func &&f, const std::set<int> &active_cells) const {
    Eigen::SparseMatrix<double> A(monomial_size(), sample_size());
    std::vector<Eigen::Triplet<double>> trips;
    // comprised of blocks of per-cell x per-col entries
    double mean_row_fill = (double)(A.rows()) / mesh().cell_count();
    double mean_col_fill = (double)(A.cols()) / mesh().cell_count();

    double pseudo_mean_cell_density = mean_row_fill * mean_col_fill;

    trips.reserve(int(pseudo_mean_cell_density * mesh().cell_count()));

    utils::loop_over_active_indices(
        _mesh.cell_count(), active_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto L = f(c);
            auto RC = c.local_to_world_monomial_indices();
            auto CC = c.local_to_world_sample_indices();
            utils::local_to_world_sparse_triplets(RC, CC, L, trips);
        });

    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}
Eigen::SparseMatrix<double> PointMomentIndexer::sample_laplacian(
    const std::set<int> &active_cells) const {
    return sample_to_sample_cell_matrix(
        [&](const PointMomentVEM2Cell &cell) -> mtao::MatXd {
            auto Pis = cell.dirichlet_projector();
            auto G = cell.monomial_dirichlet_grammian();
            auto E = cell.dirichlet_projector_error();

            auto R = Pis.transpose() * G * Pis + E.transpose() * E;
            return R;
        },
        active_cells);
}
Eigen::SparseMatrix<double> PointMomentIndexer::sample_to_poly_l2(
    const std::set<int> &active_cells) const {
    return sample_to_poly_cell_matrix(
        [&](const PointMomentVEM2Cell &cell) -> mtao::MatXd {
            auto R = cell.l2_projector();
            return R;
        },
        active_cells);
}
Eigen::SparseMatrix<double> PointMomentIndexer::sample_to_poly_dirichlet(
    const std::set<int> &active_cells) const {
    return sample_to_poly_cell_matrix(
        [&](const PointMomentVEM2Cell &cell) -> mtao::MatXd {
            auto R = cell.dirichlet_projector();
            return R;
        },
        active_cells);
}

Eigen::SparseMatrix<double> PointMomentIndexer::poly_l2_grammian(
    const std::set<int> &active_cells) const {
    return poly_to_poly_cell_matrix(
        [&](const PointMomentVEM2Cell &cell) -> mtao::MatXd {
            auto R = cell.monomial_l2_grammian();
            return R;
        },
        active_cells);
}

template <typename Func>
Eigen::SparseMatrix<double> PointMomentIndexer::sample_to_sample_cell_matrix(
    Func &&f, const std::set<int> &active_cells) const {
    Eigen::SparseMatrix<double> A(sample_size(), sample_size());
    std::vector<Eigen::Triplet<double>> trips;
    // comprised of blocks of per-cell x per-col entries
    double mean_row_fill = (double)(A.rows()) / mesh().cell_count();
    double mean_col_fill = (double)(A.cols()) / mesh().cell_count();

    double pseudo_mean_cell_density = mean_row_fill * mean_col_fill;

    trips.reserve(int(pseudo_mean_cell_density * mesh().cell_count()));

    utils::loop_over_active_indices(
        _mesh.cell_count(), active_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto L = f(c);
            auto RC = c.local_to_world_sample_indices();
            const auto &CC = RC;
            utils::local_to_world_sparse_triplets(RC, CC, L, trips);
        });

    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

// Func should be mtao::MatXd(const PointMomentIndexerCell&) where the returned
// is local_sample x local_sample shaped
template <typename Func>
Eigen::SparseMatrix<double> PointMomentIndexer::poly_to_poly_cell_matrix(
    Func &&f, const std::set<int> &active_cells) const {
    Eigen::SparseMatrix<double> A(monomial_size(), monomial_size());
    std::vector<Eigen::Triplet<double>> trips;
    // comprised of blocks of per-cell x per-col entries
    double mean_row_fill = (double)(A.rows()) / mesh().cell_count();
    double mean_col_fill = (double)(A.cols()) / mesh().cell_count();

    double pseudo_mean_cell_density = mean_row_fill * mean_col_fill;

    trips.reserve(int(pseudo_mean_cell_density * mesh().cell_count()));

    utils::loop_over_active_indices(
        _mesh.cell_count(), active_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto L = f(c);
            auto RC = c.local_to_world_monomial_indices();
            const auto &CC = RC;
            utils::local_to_world_sparse_triplets(RC, CC, L, trips);
        });

    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

PointMomentIndexer::PointMomentIndexer(const VEMMesh2 &mesh, size_t max_degree)
    : PointMomentIndexer(mesh,
                         std::vector<size_t>(mesh.cell_count(), max_degree)) {}

PointMomentIndexer::PointMomentIndexer(const VEMMesh2 &mesh,
                                       std::vector<size_t> max_degrees)
    : _mesh(mesh),
      _point_sample_indexer(
          mesh, max_edge_samples(mesh, offset_values(max_degrees, -1))),
      _moment_indexer(mesh, offset_values(max_degrees, -2)),
      _monomial_indexer(mesh, std::move(max_degrees)) {}

size_t PointMomentIndexer::sample_size() const {
    return point_sample_size() + moment_size();
}
size_t PointMomentIndexer::point_sample_size() const {
    return point_sample_indexer().num_coefficients();
}
size_t PointMomentIndexer::moment_size() const {
    return moment_indexer().num_coefficients();
}
size_t PointMomentIndexer::monomial_size() const {
    return monomial_indexer().num_coefficients();
}

/*
template <int D>
auto PointMomentIndexer::_coefficients_from_point_sample_function(
    const std::function<typename mtao::Vector<double, D>(const mtao::Vec2d &)>
        &f,
    const mtao::ColVecs2d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const -> mtao::ColVectors<double, D> {
    auto R = _homogeneous_coefficients_from_point_sample_function(
        f, P, cell_particles, active_cells);
    auto Dat = R.template topRows<D>();
    auto W = R.row(D).transpose();
    mtao::VecXd w = (W.array().abs() > 1e-10).select(1.0 / W.array(), 0.0);
    spdlog::info("coefficients_from_point_sample_functions: returning");
    return Dat * w.asDiagonal();
}
template <int D>
mtao::ColVectors<double, D + 1>
PointMomentIndexer::_homogeneous_coefficients_from_point_sample_function(
    const std::function<typename mtao::Vector<double, D>(const mtao::Vec2d &)>
        &f,
    const mtao::ColVecs2d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    mtao::ColVectors<double, D + 1> R(D + 1, sample_size());

    auto Dat = R.template topRows<D>();
    auto W = R.row(D);

#pragma omp parallel for
    for (size_t j = 0; j < point_sample_size(); ++j) {
        mtao::Vec2d p = point_sample_indexer().get_position(j);
        Dat.col(j) = f(p);
    }

    int cell_index = 0;
#pragma omp parallel for
    for (cell_index = 0; cell_index < cell_particles.size(); ++cell_index) {
        const auto &particles = cell_particles[cell_index];

        const auto &momi = moment_indexer();

        auto c = get_cell(cell_index);
        size_t mom_size = c.moment_size();
        if (mom_size == 0) {
            continue;
        }
        size_t mom_offset = c.global_moment_index_offset();

        auto pblock = Dat.block(0, mom_offset, D, mom_size);

        pblock.setZero();

        auto run = [&](auto &&pt) {
            auto v = f(pt);
            pblock +=
                v * c.evaluate_monomials_by_size(pblock.cols(), pt).transpose();
        };

        for (auto &&p : particles) {
            run(P.col(p));
        }
        auto samples = c.vertices();
        for (auto &&p : samples) {
            mtao::Vec2d pt = point_sample_indexer().get_position(p);
            run(pt);
        }

        pblock /= particles.size() + samples.size();
    }
    return R;
}

mtao::ColVecs2d PointMomentIndexer::coefficients_from_point_sample_function(
    const std::function<mtao::Vec2d(const mtao::Vec2d &)> &f,
    const mtao::ColVecs2d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    return _coefficients_from_point_sample_function(f, P, cell_particles,
                                                    active_cells);
}

mtao::VecXd PointMomentIndexer::coefficients_from_point_sample_function(
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
*/
mtao::ColVecs3d PointMomentIndexer::weighted_edge_samples(int edge_index) const {
    // degree = number of points
    auto &&[P, W] = mtao::quadrature::gauss_lobatto_data<double>(point_sample_indexer().num_edge_indices(edge_index));

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
mtao::ColVecs2d PointMomentIndexer::edge_samples(int edge_index, bool interior_only) const {

    return point_sample_indexer().edge_sample_points(edge_index,interior_only);
}

template <int D>
mtao::ColVectors<double, D + 1>
PointMomentIndexer::_homogeneous_coefficients_from_point_values(
    const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
    const std::function<double(const mtao::Vec2d &, const mtao::Vec2d &)> &rbf,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    utils::CoefficientAccumulator<PointMomentIndexer> ca(*this);
    return ca.homogeneous_coefficients_from_point_values(V, P, cell_particles,
                                                         active_cells, rbf);
}
template <int D>
mtao::ColVectors<double, D + 1>
PointMomentIndexer::_homogeneous_coefficients_from_point_sample_function(
    const std::function<typename mtao::Vector<double, D>(const mtao::Vec2d &)>
        &f,
    const mtao::ColVecs2d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    utils::CoefficientAccumulator<PointMomentIndexer> ca(*this);
    return ca.homogeneous_coefficients_from_point_function(f, P, cell_particles,
                                                           active_cells);
}

template <int D>
mtao::ColVectors<double, D>
PointMomentIndexer::_coefficients_from_point_sample_function(
    const std::function<typename mtao::Vector<double, D>(const mtao::Vec2d &)>
        &f,
    const mtao::ColVecs2d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    auto R = _homogeneous_coefficients_from_point_sample_function(
        f, P, cell_particles, active_cells);
    return utils::dehomogenize_vector_points(R);
}
template <int D>
mtao::ColVectors<double, D> PointMomentIndexer::_coefficients_from_point_values(
    const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
    const std::function<double(const mtao::Vec2d &, const mtao::Vec2d &)> &rbf,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    auto R = _homogeneous_coefficients_from_point_values(
        V, P, rbf, cell_particles, active_cells);
    return utils::dehomogenize_vector_points(R);
}

mtao::VecXd PointMomentIndexer::coefficients_from_point_sample_function(
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

mtao::ColVecs2d PointMomentIndexer::coefficients_from_point_sample_function(
    const std::function<mtao::Vec2d(const mtao::Vec2d &)> &f,
    const mtao::ColVecs2d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    return _coefficients_from_point_sample_function<2>(
        [&](const mtao::Vec2d &p) -> mtao::Vector<double, 2> { return f(p); },
        P, cell_particles, active_cells);
}
mtao::VecXd PointMomentIndexer::coefficients_from_point_values(
    const mtao::VecXd &V, const mtao::ColVecs2d &P,
    const std::function<double(const mtao::Vec2d &, const mtao::Vec2d &)> &rbf,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    return _coefficients_from_point_values<1>(V, P, rbf, cell_particles,
                                              active_cells);
}

mtao::ColVecs2d PointMomentIndexer::coefficients_from_point_values(
    const mtao::ColVecs2d &V, const mtao::ColVecs2d &P,
    const std::function<double(const mtao::Vec2d &, const mtao::Vec2d &)> &rbf,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    return _coefficients_from_point_values<2>(V, P, rbf, cell_particles,
                                              active_cells);
}

}  // namespace vem

