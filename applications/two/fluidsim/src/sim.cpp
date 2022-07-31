#include "vem/fluidsim_2d/sim.hpp"

#include "vem/fluidsim_2d/fluidvem2.hpp"
#include "vem/fluidsim_2d/flux_moment_fluidvem2.hpp"
#define USE_RBF_PARTICLE_GRID_PROJECTION

#include <eigen3/unsupported/Eigen/SparseExtra>
#include <mtao/eigen/partition_vector.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/interpolation/radial_basis_function.hpp>
#include <mtao/geometry/point_cloud/bridson_poisson_disk_sampling.hpp>
#include <vem/serialization/serialize_eigen.hpp>
#include <vem/serialization/serialize_text.hpp>
#include <vem/utils/loop_over_active.hpp>
#include <vem/utils/monomial_coefficient_projection.hpp>
#include <vem/utils/parent_maps.hpp>

namespace vem::fluidsim_2d {
Sim::Sim(const VEMMesh2& mesh, int degree,
         std::shared_ptr<serialization::Inventory> inventory)
    : FluxMomentFluidVEM2(mesh, degree),
      velocity(velocity_indexer().monomial_indexer()),
      inventory(inventory != nullptr
                    ? inventory
                    : std::make_shared<serialization::Inventory>(
                          serialization::Inventory::from_scratch(
                              "vem_fluidsim_2d", true)))
// inventory_handler(*inventory)

{
    initialize_inventory();
    sample_velocities.resize(2, velocity_stride_sample_size());
    sample_density.resize(velocity_stride_sample_size());
    resize_particles(10 * active_cell_count());
    spdlog::info("Initially {} particles", num_particles());
    initialize();
    spdlog::info("Done initializing");
}

void Sim::step(double dt) {
    spdlog::info("Taking a step {}", dt);
    spdlog::trace("Updating buoyancy");
    if (do_buoyancy) {
        update_buoyancy_particles_with_source();
    }

    spdlog::trace("Projecting particle velocities to field");
#if defined(USE_RBF_PARTICLE_GRID_PROJECTION)
    particle_to_sample(dt, .3 * mesh().dx());
#else
    update_moments_from_particles();
    particle_to_sample();
#endif
    spdlog::trace("Updating polynomial velocity");
    update_polynomial_velocity();
    update_velocity_divergence();

    auto step_inv = initialize_step_inventory();
    step_inv.add_metadata("timestep", dt);
    // TODO: cfl things
    double remaining = dt;
    update_particle_cell_cache();
    for (int j = 0; remaining > 0 && j < max_substep_count; ++j) {
        auto substep_inv = serialization::FrameInventory::for_creation(
            step_inv, j, "substep_{}");
        active_inventory = &substep_inv;
        double substep;

        {  // substep determining stuff
            double cfl_val = cfl();
            substep_inv.add_metadata("cfl", cfl_val);

            substep = std::min(remaining, cfl_val);
            // dont let substep get too low, somewhat redundant with teh last
            // for loop
            substep = std::max(dt / max_substep_count, substep);
            // when we get close to the end we just "finish the job"
            if (substep >= remaining) {
                substep_inv.add_metadata("final", true);
                substep = remaining;
                remaining = 0;
                if (j == 0) {
                    spdlog::info("Doing a full step at once (dt[{}])", dt);
                } else {
                    spdlog::info(
                        "Taking the last substep (substep[{}] / dt[{}])",
                        substep, dt);
                }
            } else {
                substep_inv.add_metadata("final", false);
                spdlog::info(
                    "Taking a substep (substep[{}] + elapseed[{}] < dt[{}]) "
                    "(i.e "
                    "remaining[{}] < dt[{}]){}",
                    substep, dt - remaining, dt, remaining, dt,
                    bool(remaining > 0));
                remaining -= substep;
            }
            substep_inv.add_metadata("substep", substep);
        }

        if (do_buoyancy) {
            update_buoyancy_particles_with_source();
        }
        // update velocity dta
        update_sample_data(substep);
        // old_velocity_coefficients = velocity.coefficients();

        if (do_buoyancy) {
            add_buoyancy_force(substep);
        }
        update_polynomial_velocity();
        // do the one grid operation
        if (do_pressure) {
            pressure_projection();
            serialization::serialize_VecXd(substep_inv, "pressure", pressure);
            serialization::serialize_VecXd(substep_inv, "velocity_divergence",
                                           velocity_divergence);
        }
        update_particle_cell_cache();

        // update_polynomial_velocity();

        if (do_advect) {
            advect(substep);
        }
        // std::cout << "Particle positions 3: " << particle_positions().norm()
        //          << std::endl;
        serialization::serialize_points4(substep_inv, "particles", particles);
        serialization::serialize_points2(substep_inv, "sample_velocities",
                                         sample_velocities);
        // step_inv.copy_asset_from_inventory(*active_inventory,
        //                                   "backpropagation_directions");
        step_inv.copy_asset_from_inventory(substep_inv, "pressure");
        step_inv.copy_asset_from_inventory(substep_inv, "velocity_divergence");
        active_inventory = nullptr;
    }
    step_inv.add_metadata("complete", true, false);
    frame_index++;
}

void Sim::resize_particles(size_t size) {
    particles.resize(4, size);
    particle_density.resize(size);
    particle_density.setZero();
}

void Sim::initialize_inventory() {
    inventory->add_metadata("type", "fluidsim");
    inventory->add_metadata("dimension", 2);
    inventory->add_metadata(
        "visualization_manifest",
        nlohmann::json::object(
            {{"velocity", "vector2_field"},
             {"velocity_x", "scalar_field"},
             {"velocity_y", "scalar_field"},
             {"particles", "point2,velocity2,density1"},
             {"mesh_velocities", "point2,velocity2"},
             {"backpropagation_directions", "point2,velocity2"},
             {"divergence", "scalar_field1"},
             {"velocity_divergence", "scalar_field1"},
             {"pressure", "scalar_field1"},
             {"density", "scalar_field"}

            }));
    // this will let me recover from bad frames!
    inventory->set_immediate_mode();
}
double Sim::cfl() const {
    double dx = mesh().dx();
    return dx / sample_velocities.colwise().norm().maxCoeff();
    double ret = std::numeric_limits<double>::max();
    for (auto&& j : active_cells()) {
        auto c = get_velocity_cell(j);
        double diameter = c.diameter();
        diameter = dx;
        size_t deg = c.monomial_degree();
        double max_vel = 0;
        for (auto&& pidx : c.local_to_world_sample_indices()) {
            double vn = sample_velocities.col(pidx).norm();
            max_vel = std::max<double>(max_vel, vn);
        }
        if (max_vel > 1e-5) {
            ret = std::min(ret, diameter / max_vel);
        }
    }
    return ret;
}
void Sim::set_active_cells(std::set<int> c) {
    spdlog::info("Setting active cells in fluid sim object to have {} cells",
                 c.size());
    FluidVEM2Base_noT::set_active_cells(std::move(c));
    velocity.set_active_cells(active_cells());

    // inventory->add_metadata("active_cells", active_cells());
    serialization::serialize_json(*inventory, "active_cells", active_cells());
    serialization::serialize_json(
        *inventory, "boundary_edges",
        velocity.boundary_intersector().boundary_edge_indices());
}

void Sim::initialize() {
    auto rbf =
        mtao::geometry::interpolation::SplineGaussianRadialBasisFunction<double,
                                                                         2>{};
    auto bb = mesh().bounding_box();
    mtao::Vec2d center = bb.center();
    auto func = [rbf, center](const mtao::Vec2d& p) -> mtao::Vec2d {
        mtao::Vec2d r =
            rbf.evaluate_grad(center, .1, p);  // * (p - center).normalized();
        if (!r.allFinite()) {
            r.setZero();
        }
        auto x = p - center;
        return mtao::Vec2d(-x.y(), x.x());
        return r;
    };
    spdlog::info("Initializing velocity field using an RBF");
    initialize(func);
    spdlog::info("Reinitializing particles");
    reinitialize_particles();
}
void Sim::reinitialize_particles() { reinitialize_particles(particles.cols()); }
void Sim::initialize(const std::function<mtao::Vec2d(const mtao::Vec2d&)>& f) {
    spdlog::info("Updating particle cell cache");
    update_particle_cell_cache();
    spdlog::info("Initializing particles");
    initialize_particles(num_particles(), f);
    spdlog::info("Updating particle cell cache");
    update_particle_cell_cache();
    spdlog::info("Using particles to assist loading function into samples");
    sample_velocities = coefficients_from_point_sample_vector_function(
        f, particle_positions(), particle_cell_cache);

    spdlog::info("Updating polynomial velocity field using samples");
    update_polynomial_velocity();
}

void Sim::initialize_particles(
    size_t size, const std::function<mtao::Vec2d(const mtao::Vec2d&)>& vel) {
    int per_cell = std::max<int>(1, size / active_cell_count());
    double rad = particle_radius_estimate_from_per_cell_count(per_cell);

    auto P = mtao::geometry::point_cloud::bridson_poisson_disk_sampling(
        mesh().bounding_box(), rad);
    std::set<int> keep;
    for (int j = 0; j < P.cols(); ++j) {
        if (is_valid_position(P.col(j))) {
            keep.emplace(j);
        }
    }
    // keep = {0};
    // P.col(0) << .9,.9;
    spdlog::info(
        "Initializing particles iwth radius {} produced a raw {} particles but "
        "we're keeping {} with {} active cells",
        rad, P.cols(), keep.size(), active_cells().size());

    resize_particles(keep.size());

    for (auto&& [idx, pidx] : mtao::iterator::enumerate(keep)) {
        auto p = particle_position(idx) = P.col(pidx);
        particle_velocity(idx) = vel(p);
    }

    particle_cell_cache.resize(cell_count());
    particle_cell_cache_dirty = true;
}
void Sim::initialize_particle(
    size_t i, const std::function<mtao::Vec2d(const mtao::Vec2d&)>& vel) {
    auto bb = mesh().bounding_box();
    auto p = particle_position(i);
    do {
        p = bb.sample();
    } while (!is_valid_position(p));

    particle_velocity(i) = vel(p);
}
void Sim::reinitialize_particles(size_t size) {
    initialize_particles(size, [&](const mtao::Vec2d& v) -> mtao::Vec2d {
        return velocity.get_vector(v);
    });
}
void Sim::update_particle_cell_cache() {
    particle_cell_cache.clear();
    particle_cell_cache.resize(mesh().cell_count());
    for (int i = 0; i < particles.cols(); ++i) {
        auto p = particle_position(i);
        int cell = mesh().get_cell(p);
        while (cell < 0 && cell >= particle_cell_cache.size()) {
            initialize_particle(i);
            cell = mesh().get_cell(p);
        }
        particle_cell_cache[cell].emplace(i);
    }
    particle_cell_cache_dirty = false;
}

serialization::FrameInventory Sim::initialize_step_inventory() const {
    auto step_inv =
        serialization::FrameInventory::for_creation(*inventory, frame_index);
    step_inv.add_metadata("complete", false);
    {
        spdlog::trace("Storing mesh velocities");
        auto P = mtao::eigen::vstack(
            mesh().V, sample_velocities.leftCols(mesh().vertex_count()));
        serialization::serialize_points4(step_inv, "mesh_velocities", P);
        auto& meta = step_inv.asset_metadata("mesh_velocities");
        meta["type"] = "point2,velocity2";
    }
    {
        spdlog::trace("Storing particle densities");
        // std::cout << particles.topRows<2>() << std::endl;
        auto P = mtao::eigen::vstack(particles, particle_density.transpose());
        // std::cout << P.topRows<2>() << std::endl;
        serialization::serialize_points5(step_inv, "particles", P);
        auto& meta = step_inv.asset_metadata("particles");
        meta["type"] = "point2,velocity2,density1";
    }
    {
        spdlog::trace("Storing mesh velocity");
        serialization::serialize_points2(step_inv, "velocity",
                                         velocity.coefficients());
        auto& meta = step_inv.asset_metadata("velocity");
        meta["type"] = "vector2_field";
    }
    {
        spdlog::trace("Storing velocity field x");
        serialization::serialize_VecXd(
            step_inv, "velocity_x", velocity.coefficients().row(0).transpose());
        auto& meta = step_inv.asset_metadata("velocity_x");
        meta["type"] = "scalar_field";
    }
    {
        spdlog::trace("Storing velocity field y");
        serialization::serialize_VecXd(
            step_inv, "velocity_y", velocity.coefficients().row(1).transpose());
        auto& meta = step_inv.asset_metadata("velocity_y");
        meta["type"] = "scalar_field";
    }

    {
        spdlog::trace("Storing divergence");
        serialization::serialize_VecXd(step_inv, "divergence",
                                       velocity_divergence_poly);
        auto& meta = step_inv.asset_metadata("divergence");
        meta["type"] = "scalar_field1";
    }
    if (pressure.size() > 0) {
        {
            spdlog::trace("Storing pressure");
            serialization::serialize_VecXd(step_inv, "pressure", pressure);
            auto& meta = step_inv.asset_metadata("pressure");
            meta["type"] = "scalar_field1";
        }
    }

    if (sample_density.size() > 0) {
        spdlog::trace("Storing sample densities");
        auto S = sample_to_poly_l2();
        mtao::VecXd D = S * sample_density;
        serialization::serialize_VecXd(step_inv, "density", D);
        auto& meta = step_inv.asset_metadata("density");
        meta["type"] = "scalar_field";
    }
    return std::move(step_inv);
}

double Sim::particle_radius_estimate_from_per_cell_count(size_t count) const {
    double side_count = std::pow<double>(count, 1.0 / 2);
    side_count = std::max<double>(side_count, 2);
    double dx = mesh().dx();
    return dx / side_count;
}

}  // namespace vem::fluidsim_2d
