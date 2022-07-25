#pragma once
#include <memory>
#if defined(VEM_USE_PYTHON)
#include <mtao/python/load_python_function.hpp>
#endif
#include <vem/creator2.hpp>
#include <vem/fluidsim_2d/sim.hpp>
#include <vem/mesh.hpp>

namespace vem::fluidsim_2d {
// you can set the mesh, set sim parameters, set boundary conditions, choose
// which cells to use, and finally run the simulation
struct SimScene {
    void set_mesh_settings(const nlohmann::json &mesh);
    VEMMesh2Creator _mesh_creator;
    // every time a mesh is created we will dump it to disk
    void create_mesh();
    const std::set<int> &active_cell_regions() const;
    void create_sim();
    void update_velocity_from_func();

    std::shared_ptr<const VEMMesh2> _mesh;
    std::shared_ptr<Sim> _sim;

    int desired_particle_count_per_cell = 20;
    int max_degree = 1;
    float timestep = .02f;

    void set_default_function();
    // set of cells used for each region
    std::vector<std::set<int>> cell_regions;
    // our currently chosen region
    std::optional<int> active_cell_region_index;
    std::shared_ptr<serialization::Inventory> inventory =
        std::make_shared<serialization::Inventory>(
            serialization::Inventory::from_scratch("2d_fluidsim"));

#if defined(VEM_USE_PYTHON)
    // maps (point) -> (vector)
    std::shared_ptr<mtao::python::PythonFunction> initial_velocity;
    std::string velocity_function;
#endif
};
}  // namespace vem::fluidsim_2d
