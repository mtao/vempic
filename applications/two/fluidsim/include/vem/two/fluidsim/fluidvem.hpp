
#pragma once
#include <Eigen/Sparse>
#include <mtao/eigen/mat_to_triplets.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <set>
#include <vem/two/fluidsim/cell.hpp>
#include <vem/mesh.hpp>
#include <vem/two/point_moment_indexer.hpp>
#include <vem/polynomials/utils.hpp>
#include <vem/utils/cell_identifier.hpp>
#include <vem/utils/local_to_world_sparse_triplets.hpp>
#include <vem/utils/loop_over_active.hpp>
#include <vem/two/parent_maps.hpp>
#include <vem/two/volumes.hpp>

#include "vem/two/fluidsim/cell.hpp"
namespace vem::two::fluidsim {

// Uses K degree velocities with K+1 degree pressures
//
// This results in a system of hte form
// < \nabla u, \nabla p > = <\nabla u, v >
// u^* \Pi^G* G^D \Pi^G = \Pi^G* diag(G^l2,D) diag(Pi^l2,2) u
template <typename IndexerType>
struct FluidVEM2Traits {};

struct FluidVEM2Base_noT {
    enum class CellWeightWeightMode : char { Unweighted, AreaWeighted };
    FluidVEM2Base_noT(const VEMMesh2 &_mesh);
    mtao::VecXd active_cell_velocity_polynomial_mask() const;

    mtao::VecXd active_cell_pressure_polynomial_mask() const;
    const VEMMesh2 &mesh() const { return _mesh; }
    std::shared_ptr<VEMMesh2 const> mesh_ptr() const { return _mesh.handle(); }
    bool is_active_cell(int index) const;
    template <typename Derived>
    bool is_valid_position(const Eigen::MatrixBase<Derived> &p,
                           int last_known = -1) const;
    const std::set<int> &active_cells() const;
    virtual void set_active_cells(std::set<int> c);
    size_t active_cell_count() const;

    // these should not be visible pubicly, just as helpers for derived classes

    size_t cell_count() const;

    const VEMMesh2 &_mesh;
    std::set<int> _active_cells;
};

template <typename Derived>
struct FluidVEM2Base : public FluidVEM2Base_noT {
    //#if defined(VEM_FLUX_MOMENT_FLUID)
    //    using SampleIndexer = FluxMomentIndexer;
    //#else
    //    using SampleIndexer = PointMomentIndexer;
    //#endif

    using Traits = FluidVEM2Traits<Derived>;
    using SampleIndexer = typename Traits::IndexerType;
    using CellType = typename Traits::CellType;
    CellType get_velocity_cell(size_t index) const {
        return CellType{velocity_indexer(), index};
    }
    CellType get_pressure_cell(size_t index) const {
        return CellType{pressure_indexer(), index};
    }
    mtao::ColVecs3d velocity_weighted_edge_samples(int edge_index) const {
        return velocity_indexer().weighted_edge_samples(edge_index);
    }

    mtao::ColVecs3d pressure_weighted_edge_samples(int edge_index) const {
        return pressure_indexer().weighted_edge_samples(edge_index);
    }
    // moments are k-2 degrees
    FluidVEM2Base(const VEMMesh2 &_mesh, size_t velocity_max_degree)
        : FluidVEM2Base_noT(_mesh),
          _velocity_indexer(_mesh, velocity_max_degree),
          _pressure_indexer(_mesh, velocity_max_degree + 1) {}

    // maps the monomial indexer from velocity indices to pressure indices
    Eigen::SparseMatrix<double> velocity_stride_to_pressure_monomial_map()
        const {
        Eigen::SparseMatrix<double> R(pressure_monomial_size(),
                                      velocity_stride_monomial_size());
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(R.rows());
        utils::loop_over_active_indices(
            mesh().cell_count(), active_cells(), [&](size_t cell_index) {
                auto pc = get_pressure_cell(cell_index);
                auto vc = get_velocity_cell(cell_index);
                auto p_l2w = pc.local_to_world_monomial_map();
                auto v_l2w = vc.local_to_world_monomial_map();

                auto A = p_l2w.leftCols(v_l2w.cols());
                auto B = v_l2w.transpose();
                Eigen::SparseMatrix<double> M = A * B;

                auto new_trips = mtao::eigen::mat_to_triplets(M);
                trips.insert(trips.end(), new_trips.begin(), new_trips.end());
            });
        R.setFromTriplets(trips.begin(), trips.end());
        return R;
    }

    // Pi^* G^* diag(M) G Pi
    Eigen::SparseMatrix<double> sample_laplacian() const {
        return _pressure_indexer.sample_laplacian(_active_cells);
    }

    Eigen::SparseMatrix<double> poly_velocity_l2_grammian() const {
        return velocity_indexer().poly_l2_grammian(active_cells());
    }

    // \Pi^G* diag(Pi^l2 D,2) u
    Eigen::SparseMatrix<double> sample_codivergence() const {
        auto P = sample_to_poly_l2();

        auto M = poly_velocity_l2_grammian();
        auto SPGrad = sample_to_poly_gradient();
        return SPGrad.transpose() *
               mtao::eigen::sparse_block_diagonal_repmats((M * P).eval(), 2);
    }

    Eigen::SparseMatrix<double> sample_to_poly_gradient() const {
        auto G = pressure_indexer().monomial_indexer().gradient();
        auto dPis = sample_to_poly_dirichlet();

        auto poly_v2p = velocity_stride_to_pressure_monomial_map();
        Eigen::SparseMatrix<double> B = poly_v2p.transpose();
        auto R = (mtao::eigen::sparse_block_diagonal_repmats(B, 2) * G * dPis)
                     .eval();
        return R;
    }

    // velocity samples -> velocity poly
    Eigen::SparseMatrix<double> sample_to_poly_l2() const {
        return _velocity_indexer.sample_to_poly_l2(_active_cells);
    }

    // pressure samples -> pressure poly
    Eigen::SparseMatrix<double> sample_to_poly_dirichlet() const {
        return _pressure_indexer.sample_to_poly_dirichlet(_active_cells);
    }

    // pressure to pressure grammian
    Eigen::SparseMatrix<double> poly_pressure_l2_grammian() const {
        return _pressure_indexer.poly_l2_grammian(_active_cells);
    }

    //// pressure to pressure grammian
    // Eigen::SparseMatrix<double> per_cell_poly_dirichlet_grammian() const;

    // std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> kkt_system(
    //    const ScalarConstraints &constraints,
    //    const std::set<int> &used_cells = {}) const;

    // uniformly sample the set of active cellls
    // returns a buffer of points and a vector from cells -> point indices
    std::tuple<mtao::ColVecs2d, std::vector<std::set<int>>> sample_active_cells(
        size_t samples_per_cell) const;

    mtao::VecXd coefficients_from_point_sample_function(
        const std::function<double(const mtao::Vec2d &)> &f) const;
    mtao::VecXd coefficients_from_point_sample_function(
        const std::function<double(const mtao::Vec2d &)> &f,
        int samples_per_cell) const;

    mtao::ColVecs2d coefficients_from_point_sample_vector_function(
        const std::function<mtao::Vec2d(const mtao::Vec2d &)> &f) const;
    mtao::ColVecs2d coefficients_from_point_sample_vector_function(
        const std::function<mtao::Vec2d(const mtao::Vec2d &)> &f,
        int samples_per_cell) const;

    mtao::VecXd coefficients_from_point_sample_function(
        const std::function<double(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles = {}) const {
        return _pressure_indexer.coefficients_from_point_sample_function(
            f, P, cell_particles);
    }
    mtao::ColVecs2d coefficients_from_point_sample_vector_function(
        const std::function<mtao::Vec2d(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles = {}) const {
        return _velocity_indexer.coefficients_from_point_sample_function(
            f, P, cell_particles);
    }

    size_t pressure_sample_size() const {
        return _pressure_indexer.sample_size();
    }

    size_t pressure_boundary_sample_size() const {
        return _pressure_indexer.boundary_sample_size();
    }
    size_t velocity_boundary_sample_size() const {
        return _velocity_indexer.flux_size();
    }
    size_t pressure_flux_size() const { return pressure_flux_size(); }
    size_t pressure_point_sample_size() const {
        return pressure_boundary_sample_size();
    }
    size_t pressure_moment_size() const {
        return _pressure_indexer.moment_size();
    }
    size_t pressure_monomial_size() const {
        return _pressure_indexer.monomial_size();
    }

    size_t velocity_sample_size() const {
        return 2 * velocity_stride_sample_size();
    }
    size_t velocity_stride_sample_size() const {
        return _velocity_indexer.sample_size();
    }
    size_t velocity_stride_flux_size() const {
        return velocity_boundary_sample_size();
    }
    size_t velocity_point_sample_size() const {
        return velocity_boundary_sample_size();
    }
    size_t velocity_stride_moment_size() const {
        return _velocity_indexer.moment_size();
    }
    size_t velocity_stride_monomial_size() const {
        return _velocity_indexer.monomial_size();
    }

    const SampleIndexer &velocity_indexer() const { return _velocity_indexer; }
    const SampleIndexer &pressure_indexer() const { return _pressure_indexer; }

   private:
    SampleIndexer _velocity_indexer;
    SampleIndexer _pressure_indexer;
};

template <typename Derived>
bool FluidVEM2Base_noT::is_valid_position(const Eigen::MatrixBase<Derived> &p,
                                          int last_known) const {
    int cell = mesh().get_cell(p, last_known);
    return is_active_cell(cell);
}

template <typename Derived>
mtao::VecXd FluidVEM2Base<Derived>::coefficients_from_point_sample_function(
    const std::function<double(const mtao::Vec2d &)> &f) const {
    double val = (double)(pressure_monomial_size()) / cell_count() + 2;
    return coefficients_from_point_sample_function(f, val * val);
}
template <typename Derived>
mtao::VecXd FluidVEM2Base<Derived>::coefficients_from_point_sample_function(
    const std::function<double(const mtao::Vec2d &)> &f,
    int samples_per_cell) const {
    auto [P, O] = sample_active_cells(samples_per_cell);
    return coefficients_from_point_sample_function(f, P, O);
}

template <typename Derived>
mtao::ColVecs2d
FluidVEM2Base<Derived>::coefficients_from_point_sample_vector_function(
    const std::function<mtao::Vec2d(const mtao::Vec2d &)> &f) const {
    double val = (double)(pressure_monomial_size()) / cell_count() + 2;
    return coefficients_from_point_sample_vector_function(f, val * val);
}

template <typename Derived>
mtao::ColVecs2d
FluidVEM2Base<Derived>::coefficients_from_point_sample_vector_function(
    const std::function<mtao::Vec2d(const mtao::Vec2d &)> &f,
    int samples_per_cell) const {
    auto [P, O] = sample_active_cells(samples_per_cell);
    return coefficients_from_point_sample_vector_function(f, P, O);
}

template <typename Derived>
std::tuple<mtao::ColVecs2d, std::vector<std::set<int>>>
FluidVEM2Base<Derived>::sample_active_cells(size_t samples_per_cell) const {
    std::vector<std::set<int>> ownerships(cell_count());
    mtao::ColVecs2d points(2, samples_per_cell * cell_count());
#pragma omp parallel for
    for (size_t idx = 0; idx < ownerships.size(); ++idx) {
        // for (auto &&[idx, own] : mtao::iterator::enumerate(ownerships)) {
        auto &own = ownerships[idx];
        if (is_active_cell(idx)) {
            auto c = get_pressure_cell(idx);
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
    }
    return {points, ownerships};
}

}  // namespace vem::two::fluidsim
