#pragma once
//#include <Eigen/CholmodSupport>
//#include <Eigen/SPQRSupport>
#include <nlohmann/json.hpp>
#include <optional>
#include "vem/three/fluidsim/fluidvem.hpp"
#include <vem/three/fluidsim/operator_cache.hpp>
#include <vem/three/monomial_field_embedder.hpp>
#include <vem/serialization/inventory.hpp>
#include <vem/serialization/prioritizing_inventory.hpp>
#define VEM_STATIC_OPERATORS const static

namespace vem::three::fluidsim {

class Sim : public FluidVEM3 {
   public:
    Sim(const VEMMesh3& vem, int degree, const std::set<int>& active_cells,
        std::shared_ptr<serialization::Inventory> inventory = nullptr);
    Sim(const VEMMesh3& vem, int degree,
        std::shared_ptr<serialization::Inventory> inventory = nullptr);
    void initialize_mesh(
        const std::function<mtao::Vec3d(const mtao::Vec3d&)>& f);
    void initialize_mesh();

    int num_particles() const { return particles.cols(); }
    void initialize_particles(
        size_t size,
        const std::function<mtao::Vec3d(const mtao::Vec3d&)>& velocity =
            [](const mtao::Vec3d& v) -> mtao::Vec3d {
            return mtao::Vec3d::Zero();
        },
        const std::function<double(const mtao::Vec3d&)>& density =
            [](const mtao::Vec3d& v) -> double { return 0.0; });
    void initialize_particle(
        size_t index,
        const std::function<mtao::Vec3d(const mtao::Vec3d&)>& velocity =
            [](const mtao::Vec3d& v) -> mtao::Vec3d {
            return mtao::Vec3d::Zero();
        },
        const std::function<double(const mtao::Vec3d&)>& density =
            [](const mtao::Vec3d& v) -> double { return 0.0; });
    // intiializes particles using the current velocity field
    void reinitialize_particles(size_t size);
    void reinitialize_particles();

    void set_particle_densities_from_func(
        const std::function<double(const mtao::Vec3d&)>& f =
            [](const mtao::Vec3d& v) -> double { return 0; });

    void set_particle_densities_source_indicator_func(
        const std::function<bool(const mtao::Vec3d&)>& f =
            [](const mtao::Vec3d& v) -> double { return false; });

    void max_particle_densities_from_func(
        const std::function<double(const mtao::Vec3d&)>& f =
            [](const mtao::Vec3d& v) -> double { return 0; });

    auto particle_positions() { return particles.topRows<3>(); }
    auto particle_velocities() { return particles.bottomRows<3>(); }
    auto particle_positions() const { return particles.topRows<3>(); }
    auto particle_velocities() const { return particles.bottomRows<3>(); }
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
    // point_constraint_matrix(const poisson_3d::ScalarConstraints&
    // constraints);
    // store simulation data
    // void save_state(const std::string& sim_storage_base_path,
    //                const std::string& frame_dir_format = "frame-{:4d}")
    //                const;

    void step(double dt);

    void set_velocity_field(
        const std::function<mtao::Vec3d(const mtao::Vec3d&)>& f);
    void set_velocity_field(const std::string& python_code);

    void set_boundary_velocity(
        const std::function<std::optional<mtao::Vec3d>(const mtao::Vec3d&)>& f);
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
    void semilagrangian_advect_fluxes(double dt);

    // solves KKT conditions of G u_{t+1} = G u_{t} - M^{-1} p(\nabla u - 0)
    // to find u_{t+1}
    void update_velocity_through_momentum_update();
    void pressure_projection();

    void add_buoyancy_force(double dt);
    void update_buoyancy_particles_with_source();
    void update_buoyancy_from_particles();

    // an optional radius can be passed if point samples also need to be
    // updated. The radius is used as the radius of an RBF
    void particles_to_samples(std::optional<double> radius = {});
    // void update_point_samples_using_particles(double radius);
    void update_fluxes_using_particles(double radius);
    void update_moments_from_particles();
    void update_particles_to_samples();
    void update_polynomial_velocity();

    // sample -> sample
    void update_divergence_from_velocity();
    // sample -> sample
    void update_pressure_from_divergence();
    // sample -> poly
    void update_pressure_gradient_from_pressure();
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
    // std::array<size_t, 3 + 3> kkt_per_dim_block_offsets() const;

    std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> kkt_system(
        double dt) const;
    void set_active_cells(std::set<int> c) override;

    // poisson_3d::ScalarConstraints boundary_conditions;

    mtao::ColVecs3d particle_field_velocities();

    mtao::ColVectors<double, 6> particles;
    mtao::VecXd particle_density;
    std::optional<double> flip_ratio = .97;

    mtao::VecXd sample_density;
    mtao::VecXd polynomial_density;
    std::optional<std::function<bool(const mtao::Vec3d&)>>
        density_indicator_func;
    void save_frame();

    mtao::VecXd pressure;
    mtao::ColVecs3d pressure_gradient;
    mtao::VecXd divergence;

    // does per-cell integration stuff + holds the velocity field itself
    mtao::ColVecs3d sample_velocities;
    mtao::ColVecs3d sample_pressure_gradient;
    MonomialVectorFieldEmbedder3 velocity;

    mtao::VecXi particle_cells;
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

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
        _qr_solver;

    //Eigen::SPQR<Eigen::SparseMatrix<double>> _spqr_solver;

    bool solver_warm = false;
    //Eigen::CholmodDecomposition<Eigen::SparseMatrix<double>> _cholmod_solver;
    bool force_static_stiffness_reconstruction = true;

    // caching required for flip difference compuatations
    mtao::ColVecs3d old_velocity_coefficients;

    bool use_semilagrangian_fluxes = false;
    bool _mesh_initialized = false;
    bool _particles_initialized = false;
    double emitter_density = 1;

    OperatorCache _operator_cache;
};
}  // namespace vem::fluidsim_3d
