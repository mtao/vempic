#define EIGEN_DONT_PARALLELIZE
#include "vem/flux_moment_indexer3.hpp"

#include <igl/random_points_on_mesh.h>

#include <mtao/geometry/interpolation/radial_basis_function.hpp>
#include <vem/utils/boundary_facets.hpp>
#include <vem/utils/local_to_world_sparse_triplets.hpp>
#include <vem/utils/loop_over_active.hpp>

#include "vem/utils/boundary_facets.hpp"
#include "vem/utils/coefficient_accumulator3.hpp"
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

std::vector<size_t> max_face_samples(const vem::VEMMesh3 &mesh,
                                     const std::vector<size_t> &o) {
    std::vector<size_t> face_degrees(mesh.face_count(), 0);
    for (auto &&[degree, fbm] :
         mtao::iterator::zip(o, mesh.cell_boundary_map)) {
        if (degree == size_t(-1)) {
            continue;
        }
        for (auto &&[face, sign] : fbm) {
            size_t &v = face_degrees.at(face);
            v = std ::max(degree, v);
        }
    }
    return face_degrees;
}
}  // namespace

namespace vem {

template <typename Func>
Eigen::SparseMatrix<double> FluxMomentIndexer3::sample_to_poly_cell_matrix(
    Func &&f, const std::set<int> &active_cells) const {
    Eigen::SparseMatrix<double> A(monomial_size(), sample_size());
    std::vector<Eigen::Triplet<double>> trips;
    std::vector<std::vector<Eigen::Triplet<double>>> per_cell_trips(
        mesh().cell_count());

    utils::loop_over_active_indices_tbb(
        _mesh.cell_count(), active_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto RC = c.local_to_world_monomial_indices();
            auto CC = c.local_to_world_sample_indices();
            auto L = f(c);
            // mtao::MatXd L(RC.size(), CC.size());
            auto &pct = per_cell_trips[cell_index];
            pct = utils::local_to_world_sparse_triplets(RC, CC, L);
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
template <typename Func>
Eigen::SparseMatrix<double> FluxMomentIndexer3::sample_to_sample_cell_matrix(
    Func &&f, const std::set<int> &active_cells) const {
    Eigen::SparseMatrix<double> A(sample_size(), sample_size());
    std::vector<Eigen::Triplet<double>> trips;

    std::vector<std::vector<Eigen::Triplet<double>>> per_cell_trips(
        mesh().cell_count());

    // using namespace std::chrono_literals;
    // spdlog::info("Taking a 2s nap");
    // std::this_thread::sleep_for(2s);

    utils::loop_over_active_indices_tbb(
        _mesh.cell_count(), active_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto RC = c.local_to_world_sample_indices();
            const auto &CC = RC;
            auto L = f(c);
            // mtao::MatXd L(RC.size(), CC.size());
            auto &pct = per_cell_trips[cell_index];
            pct = utils::local_to_world_sparse_triplets(RC, CC, L);
        });
    // spdlog::info("Taking a 30s nap");
    // std::this_thread::sleep_for(30s);

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
bool FluxMomentIndexer3::is_face_inactive(int face_index) const {
    return flux_indexer().degree(face_index) == size_t(-1);
}

// Func should be mtao::MatXd(const FluxMomentIndexer3Cell&) where the returned
// is local_sample x local_sample shaped
template <typename Func>
Eigen::SparseMatrix<double> FluxMomentIndexer3::poly_to_poly_cell_matrix(
    Func &&f, const std::set<int> &active_cells) const {
    Eigen::SparseMatrix<double> A(monomial_size(), monomial_size());
    std::vector<Eigen::Triplet<double>> trips;

    std::vector<std::vector<Eigen::Triplet<double>>> per_cell_trips(
        mesh().cell_count());
    utils::loop_over_active_indices_tbb(
        _mesh.cell_count(), active_cells, [&](size_t cell_index) {
            auto c = get_cell(cell_index);
            auto RC = c.local_to_world_monomial_indices();
            const auto &CC = RC;
            auto L = f(c);
            // mtao::MatXd L(RC.size(), CC.size());
            auto &pct = per_cell_trips[cell_index];
            pct = utils::local_to_world_sparse_triplets(RC, CC, L);
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

FluxMomentVEM3Cell FluxMomentIndexer3::get_cell(size_t index) const {
    return FluxMomentVEM3Cell(*this, index);
}

FluxMomentIndexer3::FluxMomentIndexer3(const VEMMesh3 &mesh, size_t max_degree)
    : FluxMomentIndexer3(mesh,
                         std::vector<size_t>(mesh.cell_count(), max_degree)) {}

FluxMomentIndexer3::FluxMomentIndexer3(const VEMMesh3 &mesh,
                                       std::vector<size_t> max_degrees)
    : _mesh(mesh),
      _flux_indexer(mesh,
                    offset_values(max_face_samples(mesh, max_degrees), -1)),
      _moment_indexer(mesh, offset_values(max_degrees, -2)),
      _monomial_indexer(mesh, std::move(max_degrees)) {
    spdlog::info("FluxMomentIndexer caching known cell types");
    for (int cell_index = 0; cell_index < this->mesh().cell_count();
         ++cell_index) {
        auto opt = this->mesh().cell_category(cell_index);
        if (opt) {
            int type = *opt;
            if (!_cached_l2_grammians.contains(type)) {
                auto c = get_cell(cell_index);
                _cached_l2_grammians[type] = c.monomial_l2_grammian();
            }

            if (!_cached_regularized_dirichlet_grammians.contains(type)) {
                auto c = get_cell(cell_index);
                _cached_regularized_dirichlet_grammians[type] =
                    c.regularized_monomial_dirichlet_grammian();
            }
        }
    }
}

size_t FluxMomentIndexer3::sample_size() const {
    return flux_size() + moment_size();
}
size_t FluxMomentIndexer3::flux_size() const {
    return flux_indexer().num_coefficients();
}
size_t FluxMomentIndexer3::moment_size() const {
    return moment_indexer().num_coefficients();
}
size_t FluxMomentIndexer3::monomial_size() const {
    return monomial_indexer().num_coefficients();
}

mtao::ColVecs4d FluxMomentIndexer3::weighted_face_samples_by_degree(
    int face_index, int max_degree) const {
    int samples = (max_degree + 2) * (max_degree + 2) * 10;
    return weighted_face_samples(face_index, samples);
}
mtao::ColVecs4d FluxMomentIndexer3::weighted_face_samples(
    int face_index, int sample_count) const {
    Eigen::MatrixXd B;
    Eigen::VectorXi FI;
    Eigen::MatrixXd X;
    const auto &F = mesh().triangulated_faces.at(face_index);
    igl::random_points_on_mesh(sample_count, mesh().V.transpose(),
                               F.transpose(), B, FI, X);

    mtao::ColVecs4d V(4, X.rows());
    V.topRows<3>() = X.transpose();
    V.row(3).setConstant(1);
    return V;
}
mtao::ColVecs4d FluxMomentIndexer3::weighted_face_samples(
    int face_index) const {
    return weighted_face_samples_by_degree(face_index,
                                           _flux_indexer.degree(face_index));
}

template <int D>
mtao::ColVectors<double, D + 1>
FluxMomentIndexer3::_homogeneous_coefficients_from_point_values(
    const mtao::ColVectors<double, D> &V, const mtao::ColVecs3d &P,
    const std::function<double(const mtao::Vec3d &, const mtao::Vec3d &)> &rbf,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    utils::CoefficientAccumulator3 ca(*this);
    return ca.homogeneous_coefficients_from_point_values(V, P, cell_particles,
                                                         active_cells, rbf);
}
template <int D>
mtao::ColVectors<double, D + 1>
FluxMomentIndexer3::_homogeneous_coefficients_from_point_sample_function(
    const std::function<typename mtao::Vector<double, D>(const mtao::Vec3d &)>
        &f,
    const mtao::ColVecs3d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    utils::CoefficientAccumulator3 ca(*this);
    return ca.homogeneous_coefficients_from_point_function(f, P, cell_particles,
                                                           active_cells);
}

template <int D>
mtao::ColVectors<double, D>
FluxMomentIndexer3::_coefficients_from_point_sample_function(
    const std::function<typename mtao::Vector<double, D>(const mtao::Vec3d &)>
        &f,
    const mtao::ColVecs3d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    auto R = _homogeneous_coefficients_from_point_sample_function(
        f, P, cell_particles, active_cells);
    return utils::dehomogenize_vector_points(R);
}
template <int D>
mtao::ColVectors<double, D> FluxMomentIndexer3::_coefficients_from_point_values(
    const mtao::ColVectors<double, D> &V, const mtao::ColVecs3d &P,
    const std::function<double(const mtao::Vec3d &, const mtao::Vec3d &)> &rbf,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    auto R = _homogeneous_coefficients_from_point_values(
        V, P, rbf, cell_particles, active_cells);
    return utils::dehomogenize_vector_points(R);
}

mtao::VecXd FluxMomentIndexer3::coefficients_from_point_sample_function(
    const std::function<double(const mtao::Vec3d &)> &f,
    const mtao::ColVecs3d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    return _coefficients_from_point_sample_function<1>(
               [&](const mtao::Vec3d &p) -> mtao::Vector<double, 1> {
                   return mtao::Vector<double, 1>(f(p));
               },
               P, cell_particles, active_cells)
        .transpose();
}

mtao::ColVecs3d FluxMomentIndexer3::coefficients_from_point_sample_function(
    const std::function<mtao::Vec3d(const mtao::Vec3d &)> &f,
    const mtao::ColVecs3d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    return _coefficients_from_point_sample_function<3>(
        [&](const mtao::Vec3d &p) -> mtao::Vector<double, 3> { return f(p); },
        P, cell_particles, active_cells);
}
mtao::VecXd FluxMomentIndexer3::coefficients_from_point_values(
    const mtao::VecXd &V, const mtao::ColVecs3d &P,
    const std::function<double(const mtao::Vec3d &, const mtao::Vec3d &)> &rbf,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    return _coefficients_from_point_values<1>(V, P, rbf, cell_particles,
                                              active_cells)
        .transpose();
}

mtao::ColVecs3d FluxMomentIndexer3::coefficients_from_point_values(
    const mtao::ColVecs3d &V, const mtao::ColVecs3d &P,
    const std::function<double(const mtao::Vec3d &, const mtao::Vec3d &)> &rbf,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    return _coefficients_from_point_values<3>(V, P, rbf, cell_particles,
                                              active_cells);
}

Eigen::SparseMatrix<double> FluxMomentIndexer3::sample_laplacian(
    const std::set<int> &active_cells) const {
    // auto prune = [](auto &&M) {
    //    for (int j = 0; j < M.size(); ++j) {
    //        auto &m = M(j);
    //        if (std::abs(m) < 1e-10) {
    //            m = 0;
    //        }
    //    }
    //};
    return sample_to_sample_cell_matrix(
        [&](const FluxMomentVEM3Cell &cell) -> mtao::MatXd {
            mtao::MatXd G;
            if (cell.cell_category()) {
                G = _cached_regularized_dirichlet_grammians.at(
                    *cell.cell_category());
            } else {
                G = cell.regularized_monomial_dirichlet_grammian();
            }
            auto K = cell.sample_monomial_dirichlet_grammian();

            mtao::MatXd Pis(G.cols(), K.rows());
            auto Glu = G.lu();
            for (int j = 0; j < K.rows(); ++j) {
                Pis.col(j) = Glu.solve(K.row(j).transpose());
            }

            G.row(0).setZero();

            ////
            auto B = cell.monomial_evaluation();
             return Pis.transpose() * G * Pis;
            mtao::MatXd E = B * Pis;
            E.noalias() = mtao::MatXd::Identity(E.rows(), E.cols()) - E;

            // return Pis.transpose() * G * Pis + E.transpose() * E;
            // R = E.transpose() * E;
            // R = E;
            // prune(Old);
            // ss << Old << "\n---\n";
            // prune(R);
            // ss << R << "\n===";
            // std::cout << ss.str() << std::endl;
            // R = Old;
            // return R;
        },
        active_cells);
}
Eigen::SparseMatrix<double> FluxMomentIndexer3::sample_to_poly_l2(
    const std::set<int> &active_cells) const {
    return sample_to_poly_cell_matrix(
        [&](const FluxMomentVEM3Cell &cell) -> mtao::MatXd {
            auto R = cell.l2_projector();
            return R;
        },
        active_cells);
}
Eigen::SparseMatrix<double> FluxMomentIndexer3::sample_to_poly_dirichlet(
    const std::set<int> &active_cells) const {
    return sample_to_poly_cell_matrix(
        [&](const FluxMomentVEM3Cell &cell) -> mtao::MatXd {
            auto R = cell.dirichlet_projector();
            // R = (R.array().abs() > 1e-10).select(R, 0);
            // std::cout << R << std::endl;
            return R;
        },
        active_cells);
}
Eigen::SparseMatrix<double> FluxMomentIndexer3::poly_l2_grammian(
    const std::set<int> &active_cells) const {
    return poly_to_poly_cell_matrix(
        [&](const FluxMomentVEM3Cell &cell) -> mtao::MatXd {
            if (cell.cell_category()) {
                return _cached_l2_grammians.at(*cell.cell_category());
            } else {
                return cell.monomial_l2_grammian();
            }
        },
        active_cells);
}

}  // namespace vem

