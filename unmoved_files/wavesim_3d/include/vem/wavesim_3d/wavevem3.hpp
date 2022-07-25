
#pragma once
#include <Eigen/Sparse>
#include <set>
#include <vem/flux_moment_indexer3.hpp>
#include <vem/mesh.hpp>

#include "vem/flux_moment_cell3.hpp"

namespace vem::wavesim_3d {

// Uses K degree velocities with K+1 degree pressures
//
// This results in a system of hte form
// < \nabla u, \nabla p > = <\nabla u, v >
// u^* \Pi^G* G^D \Pi^G = \Pi^G* diag(G^l2,D) diag(Pi^l2,3) u

struct WaveVEM3 {
    enum class CellWeightWeightMode : char { Unweighted, AreaWeighted };
    using WaveVEM3Cell = FluxMomentVEM3Cell;
    WaveVEM3Cell get_pressure_cell(size_t index) const;
    // moments are k-3 degrees
    WaveVEM3(const VEMMesh3 &_mesh, size_t velocity_max_degree);

    mtao::VecXd active_cell_velocity_polynomial_mask() const;

    mtao::VecXd active_cell_polynomial_mask() const;

    Eigen::SparseMatrix<double> polynomial_to_sample_evaluation_matrix(
        bool use_pressure,
        CellWeightWeightMode mode = CellWeightWeightMode::AreaWeighted) const;

    // Pi^* G^* diag(M) G Pi
    Eigen::SparseMatrix<double> sample_laplacian() const;

    // velocity samples -> velocity poly
    Eigen::SparseMatrix<double> sample_to_poly_l2() const;

    // pressure samples -> pressure poly
    Eigen::SparseMatrix<double> sample_to_poly_dirichlet() const;

    std::tuple<mtao::ColVecs3d, std::vector<std::set<int>>> sample_active_cells(
        size_t samples_per_cell) const;
    mtao::VecXd coefficients_from_point_sample_function(
        const std::function<double(const mtao::Vec3d &)> &f) const;
    mtao::VecXd coefficients_from_point_sample_function(
        const std::function<double(const mtao::Vec3d &)> &f,
        int samples_per_cell) const;
    mtao::VecXd coefficients_from_point_sample_function(
        const std::function<double(const mtao::Vec3d &)> &f,
        const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles = {}) const;

    size_t sample_size() const;
    size_t flux_size() const;
    size_t moment_size() const;
    size_t monomial_size() const;

    size_t cell_count() const;

    const VEMMesh3 &mesh() const { return _mesh; }
    std::shared_ptr<VEMMesh3 const> mesh_ptr() const { return _mesh.handle(); }

    const FluxMomentIndexer3 &indexer() const { return _indexer; }

    bool is_active_cell(int index) const;
    template <typename Derived>
    bool is_valid_position(const Eigen::MatrixBase<Derived> &p,
                           int last_known = -1) const;
    const std::set<int> &active_cells() const;
    virtual void set_active_cells(std::set<int> c);
    size_t active_cell_count() const;

   private:
    // Func should be mtao::MatXd(const WaveVEM3Cell&) where the returned is
    // poly x local_sample shaped
    template <typename Func>
    Eigen::SparseMatrix<double> sample_to_poly_cell_matrix(Func &&f) const;

    // Func should be mtao::MatXd(const WaveVEM3Cell&) where the returned is
    // local_sample x local_sample shaped
    template <typename Func>
    Eigen::SparseMatrix<double> sample_to_sample_cell_matrix(Func &&f) const;

    // Func should be mtao::MatXd(const WaveVEM3Cell&) where the returned is
    // local_poly x local_poly shaped
    template <typename Func>
    Eigen::SparseMatrix<double> poly_to_poly_cell_matrix(Func &&f) const;

    // Func should be mtao::MatXd(const WaveVEM3Cell&) where the returned is
    // poly x local_sample shaped
    template <typename Func>
    Eigen::SparseMatrix<double> poly_to_sample_cell_matrix(Func &&f) const;

   private:
    const VEMMesh3 &_mesh;
    std::set<int> _active_cells;

    FluxMomentIndexer3 _indexer;
};

template <typename Derived>
bool WaveVEM3::is_valid_position(const Eigen::MatrixBase<Derived> &p,
                                 int last_known) const {
    int cell = mesh().get_cell(p, last_known);
    return is_active_cell(cell);
}
}  // namespace vem::wavesim_3d
