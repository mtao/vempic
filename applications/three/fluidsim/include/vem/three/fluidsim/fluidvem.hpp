
#pragma once
#include <tbb/parallel_for_each.h>

#include <Eigen/Sparse>
#include <set>
#include <vem/three/flux_moment_indexer.hpp>
#include "vem/three/mesh.hpp"

#include "vem/three/fluidsim/cell.hpp"
namespace vem::three::fluidsim {

// Uses K degree velocities with K+1 degree pressures
//
// This results in a system of hte form
// < \nabla u, \nabla p > = <\nabla u, v >
// u^* \Pi^G* G^D \Pi^G = \Pi^G* diag(G^l2,D) diag(Pi^l2,3) u

struct FluidVEM3 {
    enum class CellWeightWeightMode : char { Unweighted, AreaWeighted };
    FluidVEM3Cell get_velocity_cell(size_t index) const;
    FluidVEM3Cell get_pressure_cell(size_t index) const;
    // moments are k-3 degrees
    FluidVEM3(const VEMMesh3 &_mesh, size_t velocity_max_degree);

    mtao::VecXd active_cell_velocity_polynomial_mask() const;

    mtao::VecXd active_cell_pressure_polynomial_mask() const;

    // maps the monomial indexer from velocity indices to pressure indices
    Eigen::SparseMatrix<double> velocity_stride_to_pressure_monomial_map()
        const;

    Eigen::SparseMatrix<double> polynomial_to_sample_evaluation_matrix(
        bool use_pressure,
        CellWeightWeightMode mode = CellWeightWeightMode::AreaWeighted) const;

    // Pi^* G^* diag(M) G Pi
    Eigen::SparseMatrix<double> sample_laplacian() const;

    // \Pi^G* diag(Pi^l2 D,3) u
    Eigen::SparseMatrix<double> sample_codivergence() const;

    // velocity samples -> velocity poly
    Eigen::SparseMatrix<double> sample_to_poly_l2() const;

    // pressure samples -> pressure poly
    Eigen::SparseMatrix<double> sample_to_poly_dirichlet() const;

    // pressure to pressure grammian
    Eigen::SparseMatrix<double> poly_pressure_l2_grammian() const;

    // velocity to velocity grammian
    Eigen::SparseMatrix<double> poly_velocity_l2_grammian() const;

    // P^* G V for pressure poly P, velocity poly V
    Eigen::SparseMatrix<double> sample_to_poly_codivergence() const;

    // pressure sample -> velocity sample
    Eigen::SparseMatrix<double> sample_gradient() const;

    // pressure poly -> velocity sample
    Eigen::SparseMatrix<double> sample_to_poly_gradient() const;

    //// pressure to pressure grammian
    // Eigen::SparseMatrix<double> per_cell_poly_dirichlet_grammian() const;

    // std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> kkt_system(
    //    const ScalarConstraints &constraints,
    //    const std::set<int> &used_cells = {}) const;

    mtao::ColVecs4d velocity_weighted_face_samples(int face_index) const;
    mtao::ColVecs4d pressure_weighted_face_samples(int face_index) const;
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

    mtao::ColVecs3d coefficients_from_point_sample_vector_function(
        const std::function<mtao::Vec3d(const mtao::Vec3d &)> &f) const;
    mtao::ColVecs3d coefficients_from_point_sample_vector_function(
        const std::function<mtao::Vec3d(const mtao::Vec3d &)> &f,
        int samples_per_cell) const;
    mtao::ColVecs3d coefficients_from_point_sample_vector_function(
        const std::function<mtao::Vec3d(const mtao::Vec3d &)> &f,
        const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles = {}) const;

    size_t pressure_sample_size() const;
    size_t pressure_flux_size() const;
    size_t pressure_moment_size() const;
    size_t pressure_monomial_size() const;

    size_t velocity_sample_size() const;
    size_t velocity_stride_sample_size() const;
    size_t velocity_stride_flux_size() const;
    size_t velocity_stride_moment_size() const;
    size_t velocity_stride_monomial_size() const;
    size_t cell_count() const;

    const VEMMesh3 &mesh() const { return _mesh; }
    std::shared_ptr<VEMMesh3 const> mesh_ptr() const { return _mesh.handle(); }

    const FluxMomentIndexer3 &velocity_indexer() const {
        return _velocity_indexer;
    }
    const FluxMomentIndexer3 &pressure_indexer() const {
        return _pressure_indexer;
    }

    bool is_active_cell(int index) const;
    template <typename Derived>
    bool is_valid_position(const Eigen::MatrixBase<Derived> &p,
                           int last_known = -1) const;
    template <typename Derived>
    mtao::VectorX<char> are_valid_positions(
        const Eigen::MatrixBase<Derived> &p,
        const mtao::VecXi &last_known = {}) const;
    const std::set<int> &active_cells() const;
    virtual void set_active_cells(std::set<int> c);
    size_t active_cell_count() const;

   private:
    //// Func should be mtao::MatXd(const FluidVEM3Cell&) where the returned is
    //// poly x local_sample shaped
    // template <typename Func>
    // Eigen::SparseMatrix<double> pressure_sample_to_poly_cell_matrix(
    //    Func &&f) const;

    //// Func should be mtao::MatXd(const FluidVEM3Cell&) where the returned is
    //// local_sample x local_sample shaped
    // template <typename Func>
    // Eigen::SparseMatrix<double> pressure_sample_to_sample_cell_matrix(
    //    Func &&f) const;

    //// Func should be mtao::MatXd(const FluidVEM3Cell&) where the returned is
    //// local_poly x local_poly shaped
    // template <typename Func>
    // Eigen::SparseMatrix<double> pressure_poly_to_poly_cell_matrix(
    //    Func &&f) const;

    //// Func should be mtao::MatXd(const FluidVEM3Cell&) where the returned is
    //// local_poly x local_sample shaped
    // template <typename Func>
    // Eigen::SparseMatrix<double> velocity_stride_sample_to_poly_cell_matrix(
    //    Func &&f) const;

    //// Func should be mtao::MatXd(const FluidVEM3Cell&) where the returned is
    //// poly x local_sample shaped
    // template <typename Func>
    // Eigen::SparseMatrix<double> pressure_poly_to_sample_cell_matrix(
    //    Func &&f) const;

   private:
    const VEMMesh3 &_mesh;
    std::set<int> _active_cells;

    FluxMomentIndexer3 _velocity_indexer;
    FluxMomentIndexer3 _pressure_indexer;
};

template <typename Derived>
bool FluidVEM3::is_valid_position(const Eigen::MatrixBase<Derived> &p,
                                  int last_known) const {
    int cell = mesh().get_cell(p, last_known);
    return is_active_cell(cell);
}

template <typename Derived>
mtao::VectorX<char> FluidVEM3::are_valid_positions(
    const Eigen::MatrixBase<Derived> &p, const mtao::VecXi &last_known) const {
    auto cells = mesh().get_cells(p, last_known);

    mtao::VectorX<char> ret(cells.size());
    auto zip = mtao::iterator::zip(ret, cells);
    tbb::parallel_for<int>(0, cells.size(), [&](int j) {
        ret(j) = is_active_cell(cells(j)) ? 1 : 0;
    });
    return ret;
}
}  // namespace vem::fluidsim_3d
