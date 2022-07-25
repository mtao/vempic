#include "vem/wavesim_3d/sim.hpp"
#define USE_RBF_PARTICLE_GRID_PROJECTION

#include <eigen3/unsupported/Eigen/SparseExtra>
#include <mtao/eigen/partition_vector.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/interpolation/radial_basis_function.hpp>
#include <mtao/geometry/point_cloud/bridson_poisson_disk_sampling.hpp>
#include <vem/serialization/frame_inventory.hpp>
#include <vem/serialization/serialize_eigen.hpp>
#include <vem/serialization/serialize_text.hpp>
#include <vem/serialization/serialize_vdb.hpp>
#include <vem/utils/loop_over_active.hpp>
#include <vem/utils/monomial_coefficient_projection.hpp>
#include <vem/utils/parent_maps.hpp>

namespace vem::wavesim_3d {
Sim::Sim(const VEMMesh3& mesh, int degree,
         std::shared_ptr<serialization::Inventory> inventory)
    : FluidVEM3(mesh, degree),
      velocity(velocity_indexer().monomial_indexer()),
      inventory(inventory != nullptr
                    ? inventory
                    : std::make_shared<serialization::Inventory>(
                          serialization::Inventory::from_scratch(
                              "vem_wavesim_3d", true)))
// inventory_handler(*inventory)

{
    initialize_inventory();
}

void Sim::initialize_inventory() {
    inventory->add_metadata("type", "wavesim");
    inventory->add_metadata("dimension", 3);
    inventory->add_metadata("visualization_manifest",
                            nlohmann::json::object({
                                {"pressure", "scalar_field"},
                                {"pressure_dtdt", "scalar_field"},
                            }));
    // this will let me recover from bad frames!
    inventory->set_immediate_mode();
}
double Sim::cfl() const {
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
    spdlog::info("Setting active cells in wave sim object to have {} cells",
                 c.size());
    FluidVEM3::set_active_cells(std::move(c));

    // inventory->add_metadata("active_cells", active_cells());
    // serialization::serialize_json(*inventory, "active_cells",
    // active_cells()); serialization::serialize_json(
    //    *inventory, "boundary_edges",
    //    velocity.boundary_intersector().boundary_edge_indices());
}

void Sim::initialize(const std::function<mtao::Vec3d(const mtao::Vec3d&)>& f) {
    update_particle_cell_cache();
    initialize_particles(num_particles(), f);
    spdlog::info("Using particles to assist loading function into samples");
    sample_pressure = coefficients_from_point_sample_function(
        f, particle_positions(), particle_cell_cache);
}

void Sim::step(double dt) {
    auto step_inv =
        serialization::FrameInventory::for_creation(*inventory, frame_index);
    step_inv.add_metadata("timestep", dt);
    step_inv.add_metadata("complete", false);
    update_buoyancy_particles_with_source();

    update_polynomial_pressure();
    //{
    //    auto P = mtao::eigen::vstack(
    //        mesh().V, sample_velocities.leftCols(mesh().vertex_count()));
    //    serialization::serialize_points6(step_inv, "mesh_velocities", P);
    //    auto& meta = step_inv.asset_metadata("mesh_velocities");
    //    meta["type"] = "point3,velocity3";
    //}
    //{
    //    serialization::serialize_points3(step_inv, "velocity",
    //                                     velocity.coefficients());
    //    auto& meta = step_inv.asset_metadata("velocity");
    //    meta["type"] = "vector3_field";
    //}
    //{
    //    serialization::serialize_VecXd(
    //        step_inv, "velocity_x",
    //        velocity.coefficients().row(0).transpose());
    //    auto& meta = step_inv.asset_metadata("velocity_x");
    //    meta["type"] = "scalar_field";
    //}
    //{
    //    serialization::serialize_VecXd(
    //        step_inv, "velocity_y",
    //        velocity.coefficients().row(1).transpose());
    //    auto& meta = step_inv.asset_metadata("velocity_y");
    //    meta["type"] = "scalar_field";
    //}

    //{
    //    update_velocity_divergence();
    //    serialization::serialize_VecXd(step_inv, "divergence",
    //                                   velocity_divergence_poly);
    //    auto& meta = step_inv.asset_metadata("divergence");
    //    meta["type"] = "scalar_field1";
    //}
    if (pressure.size() > 0) {
        {
            serialization::serialize_vdb(step_inv, "pressure", indexer(),
                                         pressure, dx());
            auto& meta = step_inv.asset_metadata("pressure");
            meta["type"] = "scalar_field3";
        }
    }
    wavesim_3d::semiimplicit_step(dt);

    auto substep_inv =
        serialization::FrameInventory::for_creation(step_inv, j, "substep_{}");
    active_inventory = &substep_inv;
    active_inventory = nullptr;
    step_inv.add_metadata("complete", true, false);
    frame_index++;
}

void Sim::update_polynomial_pressure() {
    pressure = sample_to_poly_dirichlet() * sample_pressure;
}

}  // namespace vem::wavesim_3d
