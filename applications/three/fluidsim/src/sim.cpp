#include "vem/fluidsim_3d/sim.hpp"
#define USE_RBF_PARTICLE_GRID_PROJECTION
#include <chrono>
#include <eigen3/unsupported/Eigen/SparseExtra>
#include <mtao/eigen/partition_vector.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/interpolation/radial_basis_function.hpp>
#include <mtao/geometry/point_cloud/bridson_poisson_disk_sampling.hpp>
#include <mtao/logging/stopwatch.hpp>
#include <thread>
#include <vem/serialization/frame_inventory.hpp>
#include <vem/serialization/serialize_eigen.hpp>
#include <vem/serialization/serialize_particles.hpp>
#include <vem/serialization/serialize_text.hpp>
//#include <vem/serialization/serialize_vdb.hpp>
#include <mtao/geometry/point_cloud/partio_loader.hpp>
#include <vem/utils/loop_over_active.hpp>
#include <vem/utils/monomial_coefficient_projection.hpp>
#include <vem/utils/parent_maps.hpp>

using namespace std::chrono_literals;
namespace vem::fluidsim_3d {
Sim::Sim(const VEMMesh3& mesh, int degree,
         std::shared_ptr<serialization::Inventory> inventory)
    : Sim(mesh, degree, {}, inventory) {}
Sim::Sim(const VEMMesh3& mesh, int degree, const std::set<int>& active_cells,
         std::shared_ptr<serialization::Inventory> inventory)
    : FluidVEM3(mesh, degree),
      velocity(velocity_indexer().monomial_indexer()),
      inventory(inventory != nullptr
                    ? inventory
                    : std::make_shared<serialization::Inventory>(
                          serialization::Inventory::from_scratch(
                              "vem_fluidsim_3d", true))),
      _operator_cache(*this, false)
// inventory_handler(*inventory)

{
    auto sw = mtao::logging::hierarchical_stopwatch("Sim::Sim");
    set_active_cells(active_cells);
    initialize_inventory();
    sample_velocities.resize(3, velocity_stride_sample_size());
    sample_density.resize(velocity_stride_sample_size());
    resize_particles(10 * active_cell_count());
    spdlog::info("Initially {} particles", num_particles());
    initialize_mesh();
    spdlog::info("Done initializing");
}
void Sim::resize_particles(size_t size) {
    particles.resize(6, size);
    particle_density.resize(size);
    particle_density.setZero();
    particle_cells.resize(size);
    particle_cells.setConstant(-1);
}

void Sim::initialize_inventory() {
    inventory->add_metadata("type", "fluidsim");
    inventory->add_metadata("dimension", 3);
    inventory->add_metadata(
        "visualization_manifest",
        nlohmann::json::object({{"velocity", "vector3_field"},
                                {"velocity_x", "scalar_field"},
                                {"velocity_y", "scalar_field"},
                                {"particles", "point3,velocity3,density1"},
                                {"mesh_velocities", "point3,velocity3"},
                                {"divergence", "scalar_field1"},
                                {"pressure", "scalar_field1"},
                                {"density", "scalar_field"}

        }));
    // this will let me recover from bad frames!
    inventory->set_immediate_mode();
}
double Sim::cfl() const {
    double dx = mesh().dx();
    {
        double max_vel = particle_velocities().colwise().norm().maxCoeff();
        return dx / max_vel;
    }

    int sample;
    double max_vel = sample_velocities.colwise().norm().maxCoeff(&sample);

    const auto& vel_idxr = velocity_indexer();
    if (sample < vel_idxr.flux_size()) {
        auto [order, index] = vel_idxr.flux_indexer().get_partition(sample);
        spdlog::info("Max sample was on face {} with center {}", index,
                     mesh().FC.col(index));
    } else {
        spdlog::error("TODO: not implemented cfl for higher order terms");
    }

    return dx / max_vel;
    double ret = std::numeric_limits<double>::max();
    for (auto&& j : active_cells()) {
        auto c = get_velocity_cell(j);
        double diameter = c.diameter();
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
    FluidVEM3::set_active_cells(std::move(c));
    velocity.set_active_cells(active_cells());

    // inventory->add_metadata("active_cells", active_cells());
    // serialization::serialize_json(*inventory, "active_cells",
    // active_cells()); serialization::serialize_json(
    //    *inventory, "boundary_edges",
    //    velocity.boundary_intersector().boundary_edge_indices());
    _operator_cache.build_cache();
}

void Sim::initialize_mesh() {
    spdlog::info("Reinitializing particles");
    sample_velocities.resize(3, velocity_stride_sample_size());
    sample_velocities.setZero();

    velocity.coefficients().resize(3, velocity_stride_monomial_size());
    velocity.coefficients().setZero();
    _mesh_initialized = true;
}
void Sim::reinitialize_particles() { reinitialize_particles(particles.cols()); }
void Sim::initialize_mesh(
    const std::function<mtao::Vec3d(const mtao::Vec3d&)>& f) {
    spdlog::info("Using particles to assist loading function into samples");

    // spdlog::info("Sleeping for 10 sec for starting mesh init");
    // std::this_thread::sleep_for(10s);
    sample_velocities = coefficients_from_point_sample_vector_function(
        f, particle_positions(), particle_cell_cache);
    // spdlog::info("Sleeping for 10 sec for finishing coeff from point sample
    // vec func"); std::this_thread::sleep_for(10s);

    spdlog::info("Updating polynomial velocity field using samples");
    update_polynomial_velocity();
    // spdlog::info("Sleeping for 2 sec for having updated poly vel");
    // std::this_thread::sleep_for(2s);
    _mesh_initialized = true;
}

void Sim::initialize_particles(
    size_t size, const std::function<mtao::Vec3d(const mtao::Vec3d&)>& vel,
    const std::function<double(const mtao::Vec3d&)>& den) {
    auto sw =
        mtao::logging::hierarchical_stopwatch("Sim::initialize_particles");
    int per_cell = std::max<int>(1, size / active_cell_count());
    double rad = particle_radius_estimate_from_per_cell_count(per_cell);

    spdlog::info("Calling bridson poisson disk sampling with radius {}", rad);
    auto P = mtao::geometry::point_cloud::bridson_poisson_disk_sampling(
        mesh().bounding_box(), rad);
    std::set<int> keep;
    for (auto&& [idx, v] : mtao::iterator::enumerate(are_valid_positions(P))) {
        if (v == 1) {
            keep.emplace(idx);
        }
    }
    // can't use bool because the proxy object might write wrong!
    /*
    std::vector<char> keep_bits(P.cols(),0);
    tbb::parallel_for<int>(0,P.cols(), [&](const int j) {
            keep_bits[j] = is_valid_position(P.col(j))?1:0;
            });
    for (int j = 0; j < P.cols(); ++j) {
        if(keep_bits[j] == 1) {
            keep.emplace(j);
        }
    }
    */
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
        particle_density(idx) = den(p);
    }

    particle_cell_cache.resize(cell_count());
    particle_cell_cache_dirty = true;
    update_particle_cell_cache();
    _particles_initialized = true;
}
void Sim::initialize_particle(
    size_t i, const std::function<mtao::Vec3d(const mtao::Vec3d&)>& vel,
    const std::function<double(const mtao::Vec3d&)>& den) {
    auto bb = mesh().bounding_box();
    auto p = particle_position(i);
    do {
        p = bb.sample();
    } while (!is_valid_position(p));

    particle_velocity(i) = vel(p);
    particle_density(i) = den(p);
}
void Sim::reinitialize_particles(size_t size) {
    auto sw =
        mtao::logging::hierarchical_stopwatch("Sim::reinitialize_particles");
    initialize_particles(
        size,
        [&](const mtao::Vec3d& v) -> mtao::Vec3d {
            return velocity.get_vector(v);
        },
        [&](const mtao::Vec3d& v) -> double {
            return velocity.Base::evaluate_monomial(v, particle_density);
        });
}
void Sim::update_particle_cell_cache() {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "Sim::update_particle_cell_cache");
    mtao::geometry::point_cloud::write_to_partio("/tmp/particles.bgeo.gz",
                                                 particle_positions());
    particle_cell_cache.clear();
    particle_cell_cache.resize(mesh().cell_count());

    auto& cells = particle_cells = mesh().get_cells(particle_positions());

    tbb::parallel_for(int(0), int(particles.cols()), [&](int i) {
        int& cell = cells(i);
        auto p = particle_position(i);
        if (cell < 0 || cell >= particle_cell_cache.size()) {
            initialize_particle(
                i,
                [&](const mtao::Vec3d& v) -> mtao::Vec3d {
                    return velocity.get_vector(v);
                },
                [&](const mtao::Vec3d& v) -> double {
                    return velocity.Base::evaluate_monomial(v,
                                                            particle_density);
                });

            cell = mesh().get_cell(p);
            spdlog::info("Particle {} moved to cell {}", i, cell);
        }
    });

    for (int i = 0; i < particles.cols(); ++i) {
        int cell = cells(i);
        particle_cell_cache[cell].emplace(i);
    }
    particle_cell_cache_dirty = false;
}
void Sim::set_particle_densities_from_func(
    const std::function<double(const mtao::Vec3d&)>& f) {
    particle_density.resize(particles.cols());
#if defined(NO_TBB_FOR)
    for (int j = 0; j < particles.cols(); ++j) {
#else
    tbb::parallel_for<int>(0, particles.cols(), [&](int j) {
#endif
        auto p = particle_position(j);
        particle_density(j) = f(p);
#if defined(NO_TBB_FOR)
    }
#else
    });
#endif
    spdlog::info("seting density func");
    density_indicator_func = f;
}

void Sim::save_frame() {
    auto sw = mtao::logging::hierarchical_stopwatch("Sim::save_frame");
    auto& step_inv = *active_inventory;
    {
        if (particle_density.size() != particles.cols()) {
            particle_density.resize(particles.cols());
            particle_density.setZero();
        }
        serialization::serialize_particles_with_velocities_and_densities(
            step_inv, "particles", particles, particle_density);
        auto& meta = step_inv.asset_metadata("particles");
        meta["type"] = "point3,velocity3,density1";
    }
    {
        std::set<int> indices;
        for (int j = 0; j < particle_density.size(); ++j) {
            if (particle_density(j) > .5) {
                indices.emplace(j);
            }
        }
        spdlog::info("{} dense particles", indices.size());
        mtao::ColVecs3d P(3, indices.size());
        for (auto&& [idx, ind] : mtao::iterator::enumerate(indices)) {
            P.col(idx) = particle_position(ind);
        }
        serialization::serialize_particles(step_inv, "dense_particles", P);
        auto& meta = step_inv.asset_metadata("dense_particles");
        meta["type"] = "point3";
    }
}

void Sim::set_particle_densities_source_indicator_func(
    const std::function<bool(const mtao::Vec3d&)>& f) {
    density_indicator_func = f;
}

void Sim::max_particle_densities_from_func(
    const std::function<double(const mtao::Vec3d&)>& f) {
    particle_density.resize(particles.cols());
#if defined(NO_TBB_FOR)
    for (int j = 0; j < particles.cols(); ++j) {
#else
    tbb::parallel_for<int>(0, particles.cols(), [&](int j) {
#endif
        auto p = particle_position(j);
        particle_density(j) = std::max(particle_density(j), f(p));
#if defined(NO_TBB_FOR)
    }
#else
    });
#endif
}
void Sim::step(double dt) {
    auto sw = mtao::logging::hierarchical_stopwatch("Sim::step");
    spdlog::info("Taking a step {}", dt);
    if (!_particles_initialized || !_mesh_initialized) {
        spdlog::error("Initialize particles [{}] and mesh [{}]",
                      _particles_initialized, _mesh_initialized);
        return;
    }
    if (!_operator_cache.built()) {
        spdlog::info("Operator cache had not been built yet, building...");
        _operator_cache.build_cache();
        spdlog::info("Done building operator cache");
    }
    using namespace std::chrono_literals;
    auto step_inv =
        serialization::FrameInventory::for_creation(*inventory, frame_index);
    active_inventory = &step_inv;

    step_inv.add_metadata("timestep", dt);
    step_inv.add_metadata("complete", false);
    save_frame();

    // spdlog::info("Sleeping for 2 sec after saving a frame");
    // std::this_thread::sleep_for(2s);
    update_particle_cell_cache();
    // spdlog::info("Sleeping for 2 sec after particle cell cache");
    // std::this_thread::sleep_for(2s);
    update_buoyancy_particles_with_source();
    // spdlog::info("Sleeping for 2 sec after updating buoyancy");
    // std::this_thread::sleep_for(2s);
    update_particles_to_samples();
    // spdlog::info("Sleeping for 2 sec after particles samples");
    // std::this_thread::sleep_for(2s);

    update_polynomial_velocity();
    // spdlog::info("Sleeping for 2 sec after poly vels");
    // std::this_thread::sleep_for(2s);
    old_velocity_coefficients = velocity.coefficients();

    // adds to sample
    add_buoyancy_force(dt);
    // spdlog::info("Sleeping for 2 sec after buoyancy force");
    // std::this_thread::sleep_for(2s);
    pressure_projection();
    // spdlog::info("Sleeping for 2 sec after pressure projection");
    // std::this_thread::sleep_for(2s);
    std::cout << "particle to field done. poly velocity norms by axis: "
              << velocity.coefficients().rowwise().norm().transpose()
              << std::endl;

    spdlog::info("Entering substepping phase");

    // TODO: cfl things
    {
        auto sw = mtao::logging::hierarchical_stopwatch("advection");
        int count = 10;
        double remaining = dt;
        for (int j = 0; remaining > 0 && j < count; ++j) {
            auto substep_inv = serialization::FrameInventory::for_creation(
                step_inv, j, "substep_{}");
            auto sw = mtao::logging::hierarchical_stopwatch(
                fmt::format("substep_{:04}", j));
            active_inventory = &substep_inv;

            double cfl_val = cfl();
            substep_inv.add_metadata("cfl", cfl_val);

            double substep = std::min(remaining, cfl_val);
            // dont let substep get too low, somewhat redundant with teh last
            // for loop
            substep = std::max(dt / count, substep);
            // when we get close to the end we just "finish the job"
            if (substep >= remaining) {
                substep_inv.add_metadata("final", true);
                substep = remaining;
                remaining = 0;
                if (count == 0) {
                    spdlog::trace("Doing a full step at once (dt[{}])", dt);
                } else {
                    spdlog::trace(
                        "Taking the last substep (substep[{}] / dt[{}])",
                        substep, dt);
                }
            } else {
                substep_inv.add_metadata("final", false);
                spdlog::trace(
                    "Taking a substep (substep[{}] + elapseed[{}] < dt[{}]) "
                    "(i.e "
                    "remaining[{}] < dt[{}]){}",
                    substep, dt - remaining, dt, remaining, dt,
                    bool(remaining > 0));
                remaining -= substep;
            }
            substep_inv.add_metadata("substep", substep);
            /*
            update_sample_velocity();
            update_buoyancy_from_particles();
            old_velocity_coefficients = velocity.coefficients();

            add_buoyancy_force(substep);
            // do the one grid operation
            pressure_projection_kkt();
            */
            // serialization::serialize_VecXd(substep_inv, "pressure",
            // pressure); serialization::serialize_VecXd(substep_inv,
            // "velocity_divergence",
            //                               velocity_divergence);

            // update_polynomial_velocity();

            advect(substep);
            // std::cout << "Particle positions 3: " <<
            // particle_positions().norm()
            //          << std::endl;
            // serialization::serialize_points4(substep_inv, "particles",
            // particles);
            // serialization::serialize_points3(substep_inv,
            // "sample_velocities",
            //                                 sample_velocities);
            active_inventory = nullptr;
        }
    }
    step_inv.add_metadata("complete", true, false);
    frame_index++;
}

// std::array<size_t, 4> Sim::kkt_per_dim_block_sizes() const {
//    size_t dimsize = velocity_sample_count() / 3;
//
//    size_t bcsize = boundary_conditions.size();
//    auto edge_face_map = vem::utils::edge_faces(mesh());
//    bcsize = 0;
//    for (auto&& [edge_idx, value] :
//         boundary_conditions.edge_integrated_flux_neumann) {
//        bcsize += edge_face_map.at(edge_idx).size();
//    }
//
//    return std::array<size_t, 4>(
//        {dimsize, dimsize, pressure_sample_count(), bcsize});
//}

// std::array<size_t, 4> Sim::kkt_block_offsets() const {
//    auto [x, y, p, l] = kkt_per_dim_block_sizes();
//    return mtao::utils::partial_sum(std::array<size_t, 3>({x + y, p, l}));
//}
// std::array<size_t, 3 + 3> Sim::kkt_per_dim_block_offsets() const {
//    return mtao::utils::partial_sum(kkt_per_dim_block_sizes());
//}

double Sim::particle_radius_estimate_from_per_cell_count(size_t count) const {
    double side_count = std::pow<double>(count, 1.0 / 3);
    side_count = std::max<double>(side_count, 3);
    double dx = mesh().dx();
    return dx / side_count;
}

}  // namespace vem::fluidsim_3d
