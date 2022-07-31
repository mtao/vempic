#pragma once
#include <Eigen/Sparse>
#include <set>
#include <vem/flux_moment_indexer.hpp>
#include <vem/mesh.hpp>

#include "vem/fluidsim_2d/fluidvem2.hpp"
#include "vem/fluidsim_2d/fluidvem2_cell.hpp"
#include "vem/flux_moment_cell.hpp"
namespace vem::fluidsim_2d {

// Uses K degree velocities with K+1 degree pressures
//
// This results in a system of hte form
// < \nabla u, \nabla p > = <\nabla u, v >
// u^* \Pi^G* G^D \Pi^G = \Pi^G* diag(G^l2,D) diag(Pi^l2,2) u

struct FluxMomentFluidVEM2;
template <>
struct FluidVEM2Traits<FluxMomentFluidVEM2> {
    using CellType = FluxMomentVEM2Cell;
    using IndexerType = FluxMomentIndexer;
};

struct FluxMomentFluidVEM2 : public FluidVEM2Base<FluxMomentFluidVEM2> {
    using Base = FluidVEM2Base<FluxMomentFluidVEM2>;
    using CellWeightWeightMode = Base::CellWeightWeightMode;
    using Base::get_pressure_cell;
    using Base::get_velocity_cell;
    // moments are k-2 degrees
    // FluxMomentFluidVEM2(const VEMMesh2 &_mesh, size_t velocity_max_degree);
    using Base::Base;

    using Base::active_cell_pressure_polynomial_mask;
    using Base::active_cell_velocity_polynomial_mask;

    // maps the monomial indexer from velocity indices to pressure indices
    using Base::velocity_stride_to_pressure_monomial_map;

    using Base::sample_laplacian;

    // \Pi^G* diag(Pi^l2 D,2) u
    using Base::sample_codivergence;

    // velocity samples -> velocity poly
    using Base::sample_to_poly_l2;

    // pressure samples -> pressure poly
    using Base::sample_to_poly_dirichlet;

    // pressure to pressure grammian
    using Base::poly_pressure_l2_grammian;

    // pressure sample -> velocity poly
    using Base::sample_to_poly_gradient;

    //// pressure to pressure grammian
    // using Base::per_cell_poly_dirichlet_grammian;

    // std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> kkt_system(
    //    const ScalarConstraints &constraints,
    //    const std::set<int> &used_cells = {}) const;

    using Base::sample_active_cells;

    /*
    using Base::coefficients_from_point_sample_function(
        const std::function<double(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles = {}) const override;

    mtao::ColVecs2d coefficients_from_point_sample_vector_function(
        const std::function<mtao::Vec2d(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles = {}) const override;*/

    using Base::pressure_moment_size;
    using Base::pressure_monomial_size;
    using Base::pressure_sample_size;

    using Base::cell_count;
    using Base::velocity_sample_size;
    using Base::velocity_stride_moment_size;
    using Base::velocity_stride_monomial_size;
    using Base::velocity_stride_sample_size;

    using Base::mesh;
    using Base::mesh_ptr;

    using Base::pressure_indexer;
    using Base::pressure_weighted_edge_samples;
    using Base::velocity_indexer;
    using Base::velocity_weighted_edge_samples;

    using Base::active_cell_count;
    using Base::active_cells;
    using Base::is_active_cell;
    using Base::is_valid_position;
    using Base::set_active_cells;
};

}  // namespace vem::fluidsim_2d
