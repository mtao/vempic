#pragma once
//#include <Eigen/CholmodSupport>
//#include <Eigen/SPQRSupport>
#include <nlohmann/json.hpp>
#include <vem/two/fluidsim/flux_moment_fluidvem.hpp>
#include "vem/two/monomial_basis_indexer.hpp"
#include <vem/two/monomial_field_embedder.hpp>
#include <vem/serialization/frame_inventory.hpp>
#include <vem/serialization/inventory.hpp>
#include <vem/serialization/prioritizing_inventory.hpp>

namespace vem::two::fluidsim {

class Sim : public FluxMomentFluidVEM2 {
   public:
    Sim(const VEMMesh2& vem, int degree,
        std::shared_ptr<serialization::Inventory> inventory = nullptr);
    void initialize(const std::function<mtao::Vec2d(const mtao::Vec2d&)>& f);
    void initialize();

    int num_particles() const { return particles.cols(); }
    void initialize_particles(
        size_t size,
        const std::function<mtao::Vec2d(const mtao::Vec2d&)>& velocity =
            [](const mtao::Vec2d& v) -> mtao::Vec2d {
            return mtao::Vec2d::Zero();
        });
    void initialize_particle(
        size_t index,
        const std::function<mtao::Vec2d(const mtao::Vec2d&)>& velocity =
            [](const mtao::Vec2d& v) -> mtao::Vec2d {
            return mtao::Vec2d::Zero();
        });
    // intiializes particles using the current velocity field
    void reinitialize_particles(size_t size);
    void reinitialize_particles();
    auto particle_positions() { return particles.topRows<2>(); }
    auto particle_velocities() { return particles.bottomRows<2>(); }
    auto particle_positions() const { return particles.topRows<2>(); }
    auto particle_velocities() const { return particles.bottomRows<2>(); }
    auto particle_position(size_t i) { return particle_positions().col(i); }
    auto particle_velocity(size_t i) { return particle_velocities().col(i); }
    auto particle_position(size_t i) const {
        return particle_positions().col(i);
    }
    auto particle_velocity(size_t i) const {
        return particle_velocities().col(i);
    }
    void resize_particles(size_t size);

    // std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd>
    // point_constraint_matrix(const poisson_2d::ScalarConstraints&
    // constraints);
    // store simulation data
    // void save_state(const std::string& sim_storage_base_path,
    //                const std::string& frame_dir_format = "frame-{:4d}")
    //                const;

    void step(double dt);
    serialization::FrameInventory initialize_step_inventory() const;

    void set_velocity_field(
        const std::function<mtao::Vec2d(const mtao::Vec2d&)>& f);
    void set_velocity_field(const std::string& python_code);

    void set_boundary_velocity(
        const std::function<std::optional<mtao::Vec2d>(const mtao::Vec2d&)>& f);
    void set_boundary_velocity(const std::string& python_code);

    // void update_particle_velocities_flip();
    // void update_particle_velocities_pic();
    void set_particle_velocities_from_grid();
    void advect_particles_with_field(double dt);
    void advect_point_samples_with_field(double dt);
    void advect_samples(double dt);
    void advect_particles(double dt);
    void advect(double dt);

    void update_particle_cell_cache();

    // solves KKT conditions of G u_{t+1} = G u_{t} - M^{-1} p(\nabla u - 0)
    // to find u_{t+1}
    void update_velocity_through_momentum_update();
    void pressure_projection();
    void update_pressure();             // from divergence
    void update_pressure_gradient();    // from pressure
    void update_velocity_divergence();  // from velocity field

    void add_buoyancy_force(double dt);
    void update_buoyancy_particles_with_source();
    void update_buoyancy_samples_from_particles();

    // an optional radius can be passed if point samples also need to be
    // updated. The radius is used as the radius of an RBF
    void particle_to_sample(double dt,
                                      std::optional<double> radius = {});
    mtao::ColVecs2d advect_points_rk2(const mtao::ColVecs2d& P, double dt);
    mtao::ColVecs2d semilagrangian_advected_edge_sample_velocities(double dt);
#if defined(VEM_FLUX_MOMENT_FLUID)
    void update_fluxes_using_particles(double radius);
    void update_fluxes_using_semilag_and_particles(double dt, double radius);
    void semilagrangian_advect_fluxes(double dt);
#else
    void update_point_samples_using_particles(double radius);
    void semilagrangian_advect_samples(double dt);
#endif

    void update_moments_from_particles();
    void update_sample_data(double dt);
    void update_polynomial_velocity();
    // cfl means (u * dt) / dx < 1
    // so dt < dx / u
    // we estimate dx with each cell's diameter and u by sample velocities
    double cfl() const;

    // estimate for the radius parameter for poisson disk sampling
    double particle_radius_estimate_from_per_cell_count(size_t count) const;

    // (sample)vector field + (monomial)pressure lambdas
    size_t system_size() const;
    // std::array<size_t, 4> kkt_per_dim_block_sizes() const;
    // std::array<size_t, 4> kkt_block_offsets() const;
    // std::array<size_t, 3 + 2> kkt_per_dim_block_offsets() const;

    std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> kkt_system(
        double dt) const;
    void set_active_cells(std::set<int> c) override;

    // std::set<int> sufficiently_marked_cells

    // poisson_2d::ScalarConstraints boundary_conditions;

    mtao::ColVecs2d particle_field_velocities();

    mtao::ColVecs4d particles;
    mtao::VecXd particle_density;

    mtao::ColVecs2d sample_velocities;
    mtao::VecXd sample_density;

    MonomialVectorFieldEmbedder velocity;
    mtao::VecXd polynomial_density;

    mtao::VecXd pressure;
    mtao::VecXd sample_pressure;

    mtao::ColVecs2d pressure_gradient;
    mtao::VecXd velocity_divergence;
    mtao::VecXd velocity_divergence_poly;

    // does per-cell integration stuff + holds the velocity field itself
    mtao::ColVecs2d sample_pressure_gradient;
    std::vector<std::set<int>> particle_cell_cache;
    bool particle_cell_cache_dirty = true;

    std::shared_ptr<serialization::Inventory> inventory;
    // yes yes, this is unsafe, but i don't want to mess with the prioritizing
    // inventory stuff at the moment
    serialization::Inventory* active_inventory = nullptr;
    // std::shared_ptr<serialization::PrioritizingInventoryHandler>
    // inventory_handler;
    int frame_index = 0;

    void initialize_inventory();

    bool static_domain = true;
    std::set<int> deactivated_pressure_samples() const;

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
        _qr_solver;

    //Eigen::SPQR<Eigen::SparseMatrix<double>> _spqr_solver;

    bool solver_warm = false;
    //Eigen::CholmodDecomposition<Eigen::SparseMatrix<double>> _cholmod_solver;
    bool force_static_stiffness_reconstruction = true;

    // caching required for flip difference compuatations
    mtao::ColVecs2d old_velocity_coefficients;
    bool use_semilagrangian_fluxes = false;
    double flip_ratio = 0.0;

    int max_substep_count = 10;
    bool do_buoyancy = true;
    bool do_advect = true;
    bool do_pressure = true;
};
}  // namespace vem::two/fluidsim
