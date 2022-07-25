#pragma once
#include <memory>
#include <mtao/python/load_python_function.hpp>
#include <vem/creator3.hpp>
#include <vem/fluidsim_3d/sim.hpp>
#include <vem/mesh.hpp>

namespace vem::fluidsim_3d {
// you can set the mesh, set sim parameters, set boundary conditions, choose
// which cells to use, and finally run the simulation
struct SimScene {
    void set_mesh_settings(const nlohmann::json &mesh);
    VEMMesh3Creator _mesh_creator;
    // every time a mesh is created we will dump it to disk
    void create_mesh();
    const std::set<int> &active_cell_regions() const;
    void create_sim(bool default_initialize = true);
    void update_pressure_from_func();

    std::shared_ptr<const VEMMesh3> _mesh;
    std::shared_ptr<Sim> _sim;

    int max_degree = 1;
    float timestep = .02f;

    void set_default_function();
    // set of cells used for each region
    std::vector<std::set<int>> cell_regions;
    // our currently chosen region
    std::optional<int> active_cell_region_index;
    std::shared_ptr<serialization::Inventory> inventory =
        std::make_shared<serialization::Inventory>(
            serialization::Inventory::from_scratch("wavesim3d_scene"));

    // maps (point) -> (vector)
    std::shared_ptr<mtao::python::PythonFunction> pressure;
    std::string pressure_function;
};
}  // namespace vem::fluidsim_2d
